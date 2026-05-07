use std::ffi::CStr;
use std::io;
use std::process::Command;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;

// ---------------------------------------------------------------------------
// Phosphor-green colour palette
// ---------------------------------------------------------------------------
const BG: Color = Color::Black;
const FG: Color = Color::Rgb(0, 255, 100); // phosphor green
const DIM: Color = Color::Rgb(0, 140, 60); // dim green for line numbers
const KEYWORD: Color = Color::Rgb(0, 255, 100); // bright green
const TYPE_COLOR: Color = Color::Cyan;
const NUMBER_COLOR: Color = Color::Yellow;
const COMMENT_COLOR: Color = Color::Rgb(0, 140, 60);
const STRING_COLOR: Color = Color::Rgb(255, 180, 80);
const DIRECTIVE_COLOR: Color = Color::Rgb(180, 120, 255);
const ERROR_COLOR: Color = Color::Rgb(255, 80, 80);
const STATUS_BG: Color = Color::Rgb(0, 40, 20);
const CURSOR_BG: Color = Color::Rgb(0, 100, 50);

// ---------------------------------------------------------------------------
// GLSL keywords and types for highlighting
// ---------------------------------------------------------------------------
const GLSL_KEYWORDS: &[&str] = &[
    "void", "return", "if", "else", "for", "while", "do", "break", "continue",
    "switch", "case", "default", "discard", "struct", "layout", "in", "out",
    "inout", "uniform", "buffer", "shared", "const", "flat", "smooth",
    "coherent", "volatile", "restrict", "readonly", "writeonly", "local_size_x",
    "local_size_y", "local_size_z", "push_constant", "set", "binding",
    "std430", "std140", "offset", "barrier", "memoryBarrier",
    "memoryBarrierShared", "memoryBarrierBuffer", "groupMemoryBarrier",
];

const GLSL_TYPES: &[&str] = &[
    "float", "double", "int", "uint", "bool",
    "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
    "uvec2", "uvec3", "uvec4", "bvec2", "bvec3", "bvec4",
    "mat2", "mat3", "mat4", "dvec2", "dvec3", "dvec4",
    "sampler2D", "sampler3D", "samplerCube", "image2D",
];

const GLSL_BUILTINS: &[&str] = &[
    "gl_GlobalInvocationID", "gl_LocalInvocationID", "gl_WorkGroupID",
    "gl_WorkGroupSize", "gl_NumWorkGroups", "gl_LocalInvocationIndex",
];

// ---------------------------------------------------------------------------
// Shader templates
// ---------------------------------------------------------------------------
const TEMPLATE_NAMES: &[&str] = &["SAXPY", "Reduction", "Image Proc", "Mandelbrot"];

fn template_saxpy() -> String {
    r#"#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer InputA {
    float a[];
};
layout(std430, binding = 1) buffer InputB {
    float b[];
};
layout(std430, binding = 2) buffer Output {
    float result[];
};

layout(push_constant) uniform PushConstants {
    float alpha;
    uint count;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < count) {
        result[idx] = alpha * a[idx] + b[idx];
    }
}
"#.to_string()
}

fn template_reduction() -> String {
    r#"#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer InputBuf {
    float data[];
};
layout(std430, binding = 1) buffer OutputBuf {
    float result[];
};

shared float sdata[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;

    sdata[tid] = data[gid];
    barrier();

    // Tree reduction in shared memory
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier();
    }

    if (tid == 0) {
        result[gl_WorkGroupID.x] = sdata[0];
    }
}
"#.to_string()
}

fn template_image_proc() -> String {
    r#"#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer InputImage {
    float pixels[];
};
layout(std430, binding = 1) buffer OutputImage {
    float out_pixels[];
};

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
};

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;
    if (pos.x >= width || pos.y >= height) return;

    uint idx = pos.y * width + pos.x;

    // 3x3 box blur
    float sum = 0.0;
    float count = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = int(pos.x) + dx;
            int ny = int(pos.y) + dy;
            if (nx >= 0 && nx < int(width) && ny >= 0 && ny < int(height)) {
                sum += pixels[uint(ny) * width + uint(nx)];
                count += 1.0;
            }
        }
    }
    out_pixels[idx] = sum / count;
}
"#.to_string()
}

fn template_mandelbrot() -> String {
    r#"#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer OutputBuf {
    float iterations[];
};

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    float cx_min;
    float cy_min;
    float cx_max;
    float cy_max;
    uint max_iter;
};

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;
    if (pos.x >= width || pos.y >= height) return;

    float x0 = cx_min + (cx_max - cx_min) * float(pos.x) / float(width);
    float y0 = cy_min + (cy_max - cy_min) * float(pos.y) / float(height);

    float x = 0.0, y = 0.0;
    uint iter = 0;
    while (x*x + y*y <= 4.0 && iter < max_iter) {
        float xtemp = x*x - y*y + x0;
        y = 2.0*x*y + y0;
        x = xtemp;
        iter++;
    }

    iterations[pos.y * width + pos.x] = float(iter) / float(max_iter);
}
"#.to_string()
}

fn get_template(index: usize) -> String {
    match index % 4 {
        0 => template_saxpy(),
        1 => template_reduction(),
        2 => template_image_proc(),
        3 => template_mandelbrot(),
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Token types for syntax highlighting
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum TokenKind {
    Keyword,
    Type,
    Builtin,
    Number,
    Comment,
    String,
    Directive,
    Punctuation,
    Plain,
}

fn style_for_token(kind: TokenKind) -> Style {
    match kind {
        TokenKind::Keyword => Style::default().fg(KEYWORD).add_modifier(Modifier::BOLD),
        TokenKind::Type => Style::default().fg(TYPE_COLOR),
        TokenKind::Builtin => Style::default().fg(TYPE_COLOR).add_modifier(Modifier::BOLD),
        TokenKind::Number => Style::default().fg(NUMBER_COLOR),
        TokenKind::Comment => Style::default().fg(COMMENT_COLOR).add_modifier(Modifier::ITALIC),
        TokenKind::String => Style::default().fg(STRING_COLOR),
        TokenKind::Directive => Style::default().fg(DIRECTIVE_COLOR),
        TokenKind::Punctuation => Style::default().fg(DIM),
        TokenKind::Plain => Style::default().fg(FG),
    }
}

fn classify_word(word: &str) -> TokenKind {
    if GLSL_KEYWORDS.contains(&word) {
        TokenKind::Keyword
    } else if GLSL_TYPES.contains(&word) {
        TokenKind::Type
    } else if GLSL_BUILTINS.contains(&word) {
        TokenKind::Builtin
    } else {
        TokenKind::Plain
    }
}

fn highlight_line(line: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Line comment
        if i + 1 < len && chars[i] == '/' && chars[i + 1] == '/' {
            let rest: String = chars[i..].iter().collect();
            spans.push(Span::styled(rest, style_for_token(TokenKind::Comment)));
            return spans;
        }

        // Preprocessor directive
        if chars[i] == '#' && spans.iter().all(|s| s.content.trim().is_empty()) {
            let rest: String = chars[i..].iter().collect();
            spans.push(Span::styled(rest, style_for_token(TokenKind::Directive)));
            return spans;
        }

        // Number literal
        if chars[i].is_ascii_digit() || (chars[i] == '.' && i + 1 < len && chars[i + 1].is_ascii_digit()) {
            let start = i;
            while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '.') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            spans.push(Span::styled(word, style_for_token(TokenKind::Number)));
            continue;
        }

        // Identifier / keyword
        if chars[i].is_ascii_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let kind = classify_word(&word);
            spans.push(Span::styled(word, style_for_token(kind)));
            continue;
        }

        // Punctuation
        if "{}()[];,=+-*/<>!&|^~%?.:" .contains(chars[i]) {
            spans.push(Span::styled(
                chars[i].to_string(),
                style_for_token(TokenKind::Punctuation),
            ));
            i += 1;
            continue;
        }

        // Whitespace and other
        let start = i;
        while i < len
            && !chars[i].is_ascii_alphanumeric()
            && chars[i] != '_'
            && chars[i] != '#'
            && chars[i] != '/'
            && !"{}()[];,=+-*/<>!&|^~%?.:".contains(chars[i])
        {
            i += 1;
        }
        let chunk: String = chars[start..i].iter().collect();
        spans.push(Span::styled(chunk, style_for_token(TokenKind::Plain)));
    }

    if spans.is_empty() {
        spans.push(Span::raw(""));
    }
    spans
}

// ---------------------------------------------------------------------------
// Convert a character index to a byte position within a string.
// If char_idx >= number of chars, returns s.len() (one past the end).
// ---------------------------------------------------------------------------
fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(byte_pos, _)| byte_pos)
        .unwrap_or(s.len())
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------
struct App {
    // Editor
    lines: Vec<String>,
    cursor_row: usize,
    cursor_col: usize,
    scroll_row: usize,
    #[allow(dead_code)]
    scroll_col: usize,

    // Output
    output_lines: Vec<(String, Style)>,
    #[allow(dead_code)]
    output_scroll: usize,

    // Template cycling
    template_index: usize,

    // Status
    status: String,
    running: bool,

    // File path for save/load
    file_path: String,

    // Dialog mode
    dialog: DialogMode,
    dialog_input: String,
}

#[derive(PartialEq)]
enum DialogMode {
    None,
    SaveAs,
    LoadFrom,
}

impl App {
    fn new() -> Self {
        let text = get_template(0);
        let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        Self {
            lines,
            cursor_row: 0,
            cursor_col: 0,
            scroll_row: 0,
            scroll_col: 0,
            output_lines: vec![
                ("gpu-shader-playground".to_string(), Style::default().fg(FG).add_modifier(Modifier::BOLD)),
                ("".to_string(), Style::default()),
                ("F5: Compile  F6: Compile+Run  F2: Save  F3: Load".to_string(), Style::default().fg(DIM)),
                ("Tab: Cycle templates  Ctrl+Q: Quit".to_string(), Style::default().fg(DIM)),
                ("".to_string(), Style::default()),
                ("Template: SAXPY loaded.".to_string(), Style::default().fg(FG)),
            ],
            output_scroll: 0,
            template_index: 0,
            status: String::from(" SAXPY | Ln 1, Col 1 "),
            running: true,
            file_path: String::from("shader.comp"),
            dialog: DialogMode::None,
            dialog_input: String::new(),
        }
    }

    fn update_status(&mut self) {
        let tmpl = TEMPLATE_NAMES[self.template_index % TEMPLATE_NAMES.len()];
        self.status = format!(
            " {} | Ln {}, Col {} | {} ",
            tmpl,
            self.cursor_row + 1,
            self.cursor_col + 1,
            self.file_path,
        );
    }

    fn ensure_cursor_visible(&mut self, editor_height: usize) {
        if editor_height == 0 {
            return;
        }
        if self.cursor_row < self.scroll_row {
            self.scroll_row = self.cursor_row;
        }
        if self.cursor_row >= self.scroll_row + editor_height {
            self.scroll_row = self.cursor_row - editor_height + 1;
        }
    }

    fn clamp_cursor(&mut self) {
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        if self.cursor_row >= self.lines.len() {
            self.cursor_row = self.lines.len() - 1;
        }
        let char_count = self.lines[self.cursor_row].chars().count();
        if self.cursor_col > char_count {
            self.cursor_col = char_count;
        }
    }

    fn push_output(&mut self, text: &str, style: Style) {
        for line in text.lines() {
            self.output_lines.push((line.to_string(), style));
        }
    }

    fn clear_output(&mut self) {
        self.output_lines.clear();
    }

    // Compile the current shader using glslangValidator
    fn compile_shader(&mut self) -> Option<String> {
        self.clear_output();
        self.push_output("Compiling...", Style::default().fg(FG));

        // Write shader to temp file
        let tmp_src = "/tmp/gpu_shader_playground.comp";
        let tmp_spv = "/tmp/gpu_shader_playground.spv";

        if let Err(e) = std::fs::write(tmp_src, self.editor_text()) {
            self.push_output(
                &format!("Error writing temp file: {}", e),
                Style::default().fg(ERROR_COLOR),
            );
            return None;
        }

        let start = Instant::now();
        let output = Command::new("glslangValidator")
            .args(["-V", "-S", "comp", "-o", tmp_spv, tmp_src])
            .output();

        let elapsed = start.elapsed();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();

                if out.status.success() {
                    self.push_output(
                        &format!("Compilation OK ({:.1}ms)", elapsed.as_secs_f64() * 1000.0),
                        Style::default().fg(FG).add_modifier(Modifier::BOLD),
                    );
                    if !stdout.trim().is_empty() {
                        self.push_output(&stdout, Style::default().fg(DIM));
                    }
                    // Report SPIR-V size
                    if let Ok(meta) = std::fs::metadata(tmp_spv) {
                        self.push_output(
                            &format!("SPIR-V: {} bytes", meta.len()),
                            Style::default().fg(FG),
                        );
                    }
                    Some(tmp_spv.to_string())
                } else {
                    self.push_output("Compilation FAILED", Style::default().fg(ERROR_COLOR).add_modifier(Modifier::BOLD));
                    if !stdout.trim().is_empty() {
                        // Parse and display errors
                        for line in stdout.lines() {
                            if line.contains("ERROR") {
                                self.push_output(line, Style::default().fg(ERROR_COLOR));
                            } else if line.contains("WARNING") {
                                self.push_output(line, Style::default().fg(NUMBER_COLOR));
                            } else {
                                self.push_output(line, Style::default().fg(DIM));
                            }
                        }
                    }
                    if !stderr.trim().is_empty() {
                        self.push_output(&stderr, Style::default().fg(ERROR_COLOR));
                    }
                    None
                }
            }
            Err(e) => {
                self.push_output(
                    &format!("Failed to run glslangValidator: {}", e),
                    Style::default().fg(ERROR_COLOR),
                );
                self.push_output(
                    "Is glslangValidator installed?",
                    Style::default().fg(DIM),
                );
                None
            }
        }
    }

    fn compile_and_run(&mut self) {
        let spv_path = self.compile_shader();

        if let Some(spv_path) = spv_path {
            self.push_output("", Style::default());
            self.push_output(
                "Dispatching on Vulkan...",
                Style::default().fg(FG).add_modifier(Modifier::BOLD),
            );

            match run_vulkan_compute(&spv_path) {
                Ok(result) => {
                    self.push_output(
                        &format!("Dispatch OK ({:.3}ms GPU time)", result.gpu_time_ms),
                        Style::default().fg(FG).add_modifier(Modifier::BOLD),
                    );
                    self.push_output(
                        &format!("Device: {}", result.device_name),
                        Style::default().fg(DIM),
                    );
                    self.push_output("", Style::default());
                    self.push_output("Output buffer (first 64 values):", Style::default().fg(FG));

                    // Display results in rows of 8
                    for chunk in result.output_data.chunks(8) {
                        let vals: Vec<String> = chunk.iter().map(|v| format!("{:10.4}", v)).collect();
                        self.push_output(&vals.join(" "), Style::default().fg(NUMBER_COLOR));
                    }
                }
                Err(e) => {
                    self.push_output(
                        &format!("Vulkan execution error: {}", e),
                        Style::default().fg(ERROR_COLOR),
                    );
                }
            }
        }
    }

    fn editor_text(&self) -> String {
        self.lines.join("\n")
    }

    fn save_file(&mut self, path: &str) {
        let text = self.editor_text();
        match std::fs::write(path, &text) {
            Ok(_) => {
                self.file_path = path.to_string();
                self.push_output(
                    &format!("Saved to {}", path),
                    Style::default().fg(FG),
                );
            }
            Err(e) => {
                self.push_output(
                    &format!("Save error: {}", e),
                    Style::default().fg(ERROR_COLOR),
                );
            }
        }
    }

    fn load_file(&mut self, path: &str) {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                self.lines = text.lines().map(|l| l.to_string()).collect();
                if self.lines.is_empty() {
                    self.lines.push(String::new());
                }
                self.cursor_row = 0;
                self.cursor_col = 0;
                self.scroll_row = 0;
                self.file_path = path.to_string();
                self.clear_output();
                self.push_output(
                    &format!("Loaded {} ({} lines)", path, self.lines.len()),
                    Style::default().fg(FG),
                );
            }
            Err(e) => {
                self.push_output(
                    &format!("Load error: {}", e),
                    Style::default().fg(ERROR_COLOR),
                );
            }
        }
    }

    fn cycle_template(&mut self) {
        self.template_index = (self.template_index + 1) % TEMPLATE_NAMES.len();
        let text = get_template(self.template_index);
        self.lines = text.lines().map(|l| l.to_string()).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.cursor_row = 0;
        self.cursor_col = 0;
        self.scroll_row = 0;
        self.clear_output();
        self.push_output(
            &format!("Template: {} loaded.", TEMPLATE_NAMES[self.template_index]),
            Style::default().fg(FG),
        );
    }

    // Insert a character at cursor
    fn insert_char(&mut self, c: char) {
        let line = &mut self.lines[self.cursor_row];
        let char_count = line.chars().count();
        if self.cursor_col > char_count {
            self.cursor_col = char_count;
        }
        let byte_pos = char_to_byte(line, self.cursor_col);
        line.insert(byte_pos, c);
        self.cursor_col += 1;
    }

    // Insert a newline at cursor
    fn insert_newline(&mut self) {
        let line = &mut self.lines[self.cursor_row];
        let byte_pos = char_to_byte(line, self.cursor_col);
        let rest = line[byte_pos..].to_string();
        line.truncate(byte_pos);

        // Auto-indent: copy leading whitespace from current line
        let indent: String = self.lines[self.cursor_row]
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect();

        let new_line = format!("{}{}", indent, rest);
        self.cursor_row += 1;
        self.lines.insert(self.cursor_row, new_line);
        self.cursor_col = indent.chars().count();
    }

    fn backspace(&mut self) {
        if self.cursor_col > 0 {
            let line = &mut self.lines[self.cursor_row];
            let char_count = line.chars().count();
            if self.cursor_col <= char_count {
                let byte_pos = char_to_byte(line, self.cursor_col - 1);
                line.remove(byte_pos);
            }
            self.cursor_col -= 1;
        } else if self.cursor_row > 0 {
            // Merge with previous line
            let current = self.lines.remove(self.cursor_row);
            self.cursor_row -= 1;
            self.cursor_col = self.lines[self.cursor_row].chars().count();
            self.lines[self.cursor_row].push_str(&current);
        }
    }

    fn delete(&mut self) {
        let char_count = self.lines[self.cursor_row].chars().count();
        if self.cursor_col < char_count {
            let byte_pos = char_to_byte(&self.lines[self.cursor_row], self.cursor_col);
            self.lines[self.cursor_row].remove(byte_pos);
        } else if self.cursor_row + 1 < self.lines.len() {
            let next = self.lines.remove(self.cursor_row + 1);
            self.lines[self.cursor_row].push_str(&next);
        }
    }
}

// ---------------------------------------------------------------------------
// Vulkan compute execution
// ---------------------------------------------------------------------------
struct ComputeResult {
    device_name: String,
    gpu_time_ms: f64,
    output_data: Vec<f32>,
}

unsafe fn find_memory_type(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
    type_filter: u32,
    properties: ash::vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mem_props = instance.get_physical_device_memory_properties(physical_device);
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && (mem_props.memory_types[i as usize].property_flags & properties) == properties
        {
            return Some(i);
        }
    }
    None
}

fn run_vulkan_compute(spv_path: &str) -> Result<ComputeResult, String> {
    let spv_bytes = std::fs::read(spv_path).map_err(|e| format!("Read SPIR-V: {}", e))?;
    if spv_bytes.len() % 4 != 0 {
        return Err("SPIR-V file size not aligned to 4 bytes".to_string());
    }

    let spv_code: Vec<u32> = spv_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    unsafe {
        // Create Vulkan instance
        let app_name = c"gpu-shader-playground";
        let engine_name = c"none";

        let app_info = ash::vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(ash::vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(ash::vk::make_api_version(0, 1, 0, 0))
            .api_version(ash::vk::make_api_version(0, 1, 2, 0));

        let create_info = ash::vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let entry = ash::Entry::load().map_err(|e| format!("Load Vulkan: {:?}", e))?;
        let instance = entry
            .create_instance(&create_info, None)
            .map_err(|e| format!("Create instance: {:?}", e))?;

        // Pick physical device
        let phys_devices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("Enumerate devices: {:?}", e))?;

        if phys_devices.is_empty() {
            instance.destroy_instance(None);
            return Err("No Vulkan devices found".to_string());
        }

        let physical_device = phys_devices[0];
        let dev_props = instance.get_physical_device_properties(physical_device);
        let device_name = CStr::from_ptr(dev_props.device_name.as_ptr())
            .to_string_lossy()
            .to_string();

        // Find compute queue family
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        let compute_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(ash::vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| "No compute queue family".to_string())?;

        // Check timestamp support
        let timestamp_period = dev_props.limits.timestamp_period;
        let timestamp_valid = queue_families[compute_family as usize].timestamp_valid_bits > 0;

        // Create logical device
        let queue_priority = [1.0f32];
        let queue_create_info = ash::vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_family)
            .queue_priorities(&queue_priority);

        let queue_create_infos = [queue_create_info];
        let device_create_info = ash::vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos);

        let device = instance
            .create_device(physical_device, &device_create_info, None)
            .map_err(|e| format!("Create device: {:?}", e))?;

        let queue = device.get_device_queue(compute_family, 0);

        // Create shader module
        let shader_create_info = ash::vk::ShaderModuleCreateInfo::default()
            .code(&spv_code);

        let shader_module = device
            .create_shader_module(&shader_create_info, None)
            .map_err(|e| {
                device.destroy_device(None);
                instance.destroy_instance(None);
                format!("Create shader module: {:?}", e)
            })?;

        // Constants
        let element_count: u32 = 256;
        let buffer_size = (element_count as u64) * std::mem::size_of::<f32>() as u64;

        // Create buffers (input + output)
        let create_buffer = |size: u64, usage: ash::vk::BufferUsageFlags| -> Result<(ash::vk::Buffer, ash::vk::DeviceMemory), String> {
            let buf_info = ash::vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

            let buffer = device
                .create_buffer(&buf_info, None)
                .map_err(|e| format!("Create buffer: {:?}", e))?;

            let mem_reqs = device.get_buffer_memory_requirements(buffer);
            let mem_type = find_memory_type(
                &instance,
                physical_device,
                mem_reqs.memory_type_bits,
                ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or_else(|| "No suitable memory type".to_string())?;

            let alloc_info = ash::vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type);

            let memory = device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| format!("Allocate memory: {:?}", e))?;

            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| format!("Bind buffer: {:?}", e))?;

            Ok((buffer, memory))
        };

        let usage_flags = ash::vk::BufferUsageFlags::STORAGE_BUFFER;
        let (input_buf, input_mem) = create_buffer(buffer_size, usage_flags).map_err(|e| {
            device.destroy_shader_module(shader_module, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
            e
        })?;
        let (output_buf, output_mem) = create_buffer(buffer_size, usage_flags).map_err(|e| {
            device.free_memory(input_mem, None);
            device.destroy_buffer(input_buf, None);
            device.destroy_shader_module(shader_module, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
            e
        })?;

        // Initialize input buffer with test data
        {
            let ptr = device
                .map_memory(input_mem, 0, buffer_size, ash::vk::MemoryMapFlags::empty())
                .map_err(|e| format!("Map input memory: {:?}", e))?
                as *mut f32;

            for i in 0..element_count {
                *ptr.add(i as usize) = i as f32;
            }
            device.unmap_memory(input_mem);
        }

        // Initialize output buffer to zero
        {
            let ptr = device
                .map_memory(output_mem, 0, buffer_size, ash::vk::MemoryMapFlags::empty())
                .map_err(|e| format!("Map output memory: {:?}", e))?
                as *mut f32;

            for i in 0..element_count {
                *ptr.add(i as usize) = 0.0;
            }
            device.unmap_memory(output_mem);
        }

        // Run the pipeline setup, dispatch, and readback in a helper closure
        // so that all Vulkan resources are cleaned up on both success and error.
        let exec_result: Result<(f64, Vec<f32>), String> = (|| -> Result<(f64, Vec<f32>), String> {
            // Create descriptor set layout
            let bindings = [
                ash::vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
                ash::vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
            ];

            let layout_info = ash::vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings);

            let desc_layout = device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| format!("Create desc layout: {:?}", e))?;

            // Create pipeline layout
            let set_layouts = [desc_layout];
            let pipeline_layout_info = ash::vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts);

            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| {
                    device.destroy_descriptor_set_layout(desc_layout, None);
                    format!("Create pipeline layout: {:?}", e)
                })?;

            // Create compute pipeline
            let entry_point = c"main";
            let stage_info = ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(entry_point);

            let pipeline_info = ash::vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);

            let pipelines = device
                .create_compute_pipelines(ash::vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(desc_layout, None);
                    format!("Create compute pipeline: {:?}", e)
                })?;

            let pipeline = pipelines[0];

            // Create descriptor pool and set
            let pool_size = ash::vk::DescriptorPoolSize::default()
                .ty(ash::vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2);

            let pool_sizes = [pool_size];
            let pool_info = ash::vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);

            let desc_pool = device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(desc_layout, None);
                    format!("Create desc pool: {:?}", e)
                })?;

            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&set_layouts);

            let desc_sets = device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| {
                    device.destroy_descriptor_pool(desc_pool, None);
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(desc_layout, None);
                    format!("Allocate desc set: {:?}", e)
                })?;

            let desc_set = desc_sets[0];

            // Update descriptor set
            let input_buf_info = ash::vk::DescriptorBufferInfo::default()
                .buffer(input_buf)
                .offset(0)
                .range(buffer_size);

            let output_buf_info = ash::vk::DescriptorBufferInfo::default()
                .buffer(output_buf)
                .offset(0)
                .range(buffer_size);

            let input_buf_infos = [input_buf_info];
            let output_buf_infos = [output_buf_info];

            let writes = [
                ash::vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&input_buf_infos),
                ash::vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(1)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&output_buf_infos),
            ];

            device.update_descriptor_sets(&writes, &[]);

            // Create timestamp query pool
            let query_pool = if timestamp_valid {
                let query_pool_info = ash::vk::QueryPoolCreateInfo::default()
                    .query_type(ash::vk::QueryType::TIMESTAMP)
                    .query_count(2);

                device.create_query_pool(&query_pool_info, None).ok()
            } else {
                None
            };

            // Macro-like closure to destroy all pipeline resources
            let cleanup_pipeline = |device: &ash::Device| {
                if let Some(qp) = query_pool {
                    device.destroy_query_pool(qp, None);
                }
                device.destroy_descriptor_pool(desc_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(desc_layout, None);
            };

            // Create command pool and buffer
            let cmd_pool_info = ash::vk::CommandPoolCreateInfo::default()
                .queue_family_index(compute_family);

            let cmd_pool = device
                .create_command_pool(&cmd_pool_info, None)
                .map_err(|e| {
                    cleanup_pipeline(&device);
                    format!("Create cmd pool: {:?}", e)
                })?;

            // From here, cleanup includes cmd_pool
            let cleanup_all = |device: &ash::Device| {
                device.destroy_command_pool(cmd_pool, None);
                cleanup_pipeline(device);
            };

            let cmd_alloc_info = ash::vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(ash::vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd_bufs = device
                .allocate_command_buffers(&cmd_alloc_info)
                .map_err(|e| {
                    cleanup_all(&device);
                    format!("Allocate cmd buf: {:?}", e)
                })?;

            let cmd_buf = cmd_bufs[0];

            // Record commands
            let begin_info = ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device
                .begin_command_buffer(cmd_buf, &begin_info)
                .map_err(|e| {
                    cleanup_all(&device);
                    format!("Begin cmd buf: {:?}", e)
                })?;

            if let Some(qp) = query_pool {
                device.cmd_reset_query_pool(cmd_buf, qp, 0, 2);
                device.cmd_write_timestamp(
                    cmd_buf,
                    ash::vk::PipelineStageFlags::TOP_OF_PIPE,
                    qp,
                    0,
                );
            }

            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                ash::vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[desc_set],
                &[],
            );

            // Dispatch 1 workgroup of 256 threads
            device.cmd_dispatch(cmd_buf, 1, 1, 1);

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(
                    cmd_buf,
                    ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    qp,
                    1,
                );
            }

            device
                .end_command_buffer(cmd_buf)
                .map_err(|e| {
                    cleanup_all(&device);
                    format!("End cmd buf: {:?}", e)
                })?;

            // Submit and wait
            let cmd_bufs_submit = [cmd_buf];
            let submit_info = ash::vk::SubmitInfo::default()
                .command_buffers(&cmd_bufs_submit);

            let wall_start = Instant::now();

            device
                .queue_submit(queue, &[submit_info], ash::vk::Fence::null())
                .map_err(|e| {
                    cleanup_all(&device);
                    format!("Queue submit: {:?}", e)
                })?;

            device
                .queue_wait_idle(queue)
                .map_err(|e| {
                    cleanup_all(&device);
                    format!("Queue wait: {:?}", e)
                })?;

            let wall_elapsed = wall_start.elapsed();

            // Read timestamp results
            let gpu_time_ms = if let Some(qp) = query_pool {
                let mut timestamps = [0u64; 2];
                let result = device.get_query_pool_results(
                    qp,
                    0,
                    &mut timestamps,
                    ash::vk::QueryResultFlags::TYPE_64,
                );
                if result.is_ok() && timestamps[1] > timestamps[0] {
                    let ticks = timestamps[1] - timestamps[0];
                    (ticks as f64) * (timestamp_period as f64) / 1_000_000.0
                } else {
                    wall_elapsed.as_secs_f64() * 1000.0
                }
            } else {
                wall_elapsed.as_secs_f64() * 1000.0
            };

            // Read back output buffer
            let mut output_data = vec![0.0f32; 64.min(element_count as usize)];
            {
                let ptr = device
                    .map_memory(output_mem, 0, buffer_size, ash::vk::MemoryMapFlags::empty())
                    .map_err(|e| {
                        cleanup_all(&device);
                        format!("Map output for read: {:?}", e)
                    })?
                    as *const f32;

                for (i, val) in output_data.iter_mut().enumerate() {
                    *val = *ptr.add(i);
                }
                device.unmap_memory(output_mem);
            }

            // Cleanup pipeline resources on success path too
            cleanup_all(&device);

            Ok((gpu_time_ms, output_data))
        })();

        // Always clean up buffers, shader, device, and instance regardless of success/error
        device.free_memory(output_mem, None);
        device.destroy_buffer(output_buf, None);
        device.free_memory(input_mem, None);
        device.destroy_buffer(input_buf, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_device(None);
        instance.destroy_instance(None);

        let (gpu_time_ms, output_data) = exec_result?;

        Ok(ComputeResult {
            device_name,
            gpu_time_ms,
            output_data,
        })
    }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------
fn draw(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &App) {
    terminal
        .draw(|frame| {
            let size = frame.area();

            // Main layout: editor left, output right, status bar bottom
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(3),
                    Constraint::Length(1), // status bar
                ])
                .split(size);

            let body = main_chunks[0];
            let status_area = main_chunks[1];

            let panes = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(60),
                    Constraint::Percentage(40),
                ])
                .split(body);

            let editor_area = panes[0];
            let output_area = panes[1];

            // Draw editor
            draw_editor(frame, app, editor_area);

            // Draw output
            draw_output(frame, app, output_area);

            // Draw status bar
            let status_line = Line::from(vec![
                Span::styled(&app.status, Style::default().fg(FG).bg(STATUS_BG)),
            ]);
            let status_widget = Paragraph::new(status_line)
                .style(Style::default().bg(STATUS_BG));
            frame.render_widget(status_widget, status_area);

            // Draw dialog if active
            if app.dialog != DialogMode::None {
                draw_dialog(frame, app, size);
            }
        })
        .ok();
}

fn draw_editor(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Editor ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM))
        .style(Style::default().bg(BG));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width < 2 || inner.height < 1 {
        return;
    }

    let line_num_width = format!("{}", app.lines.len()).len().max(3) + 1;
    let visible_rows = inner.height as usize;

    let mut lines_to_render: Vec<Line> = Vec::with_capacity(visible_rows);

    for row_idx in 0..visible_rows {
        let line_idx = app.scroll_row + row_idx;

        if line_idx >= app.lines.len() {
            // Past end of file — show tilde
            let spans = vec![
                Span::styled(
                    format!("{:>width$} ", "~", width = line_num_width - 1),
                    Style::default().fg(DIM),
                ),
            ];
            lines_to_render.push(Line::from(spans));
            continue;
        }

        let line_text = &app.lines[line_idx];

        // Line number
        let line_num = format!("{:>width$} ", line_idx + 1, width = line_num_width - 1);
        let mut spans = vec![
            Span::styled(line_num, Style::default().fg(DIM)),
        ];

        // Highlighted text
        let mut highlighted = highlight_line(line_text);

        // If this is the cursor line, we need to insert cursor highlighting
        if line_idx == app.cursor_row {
            // Rebuild spans with cursor position marked
            let mut result_spans: Vec<Span<'static>> = Vec::new();
            let mut col: usize = 0; // tracks char offset

            for span in highlighted.drain(..) {
                let span_text: &str = &span.content;
                let span_style = span.style;
                let span_chars = span_text.chars().count();

                if col + span_chars <= app.cursor_col || col > app.cursor_col {
                    // Cursor not in this span
                    result_spans.push(Span::styled(span_text.to_string(), span_style));
                } else {
                    // Cursor is within this span — split it
                    let char_offset = app.cursor_col - col;
                    let byte_offset = char_to_byte(span_text, char_offset);
                    if byte_offset > 0 {
                        result_spans.push(Span::styled(
                            span_text[..byte_offset].to_string(),
                            span_style,
                        ));
                    }
                    // Cursor character
                    let next_byte = char_to_byte(span_text, char_offset + 1);
                    if byte_offset < span_text.len() {
                        result_spans.push(Span::styled(
                            span_text[byte_offset..next_byte].to_string(),
                            span_style.bg(CURSOR_BG),
                        ));
                        if next_byte < span_text.len() {
                            result_spans.push(Span::styled(
                                span_text[next_byte..].to_string(),
                                span_style,
                            ));
                        }
                    }
                }
                col += span_chars;
            }

            // If cursor is at end of line, show a cursor block
            if app.cursor_col >= line_text.chars().count() {
                result_spans.push(Span::styled(
                    " ".to_string(),
                    Style::default().bg(CURSOR_BG),
                ));
            }

            spans.extend(result_spans);
        } else {
            spans.extend(highlighted);
        }

        lines_to_render.push(Line::from(spans));
    }

    let paragraph = Paragraph::new(lines_to_render).style(Style::default().bg(BG));
    frame.render_widget(paragraph, inner);
}

fn draw_output(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Output ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM))
        .style(Style::default().bg(BG));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let visible_rows = inner.height as usize;
    let total = app.output_lines.len();

    // Auto-scroll to bottom
    let scroll = if total > visible_rows {
        total - visible_rows
    } else {
        0
    };

    let lines: Vec<Line> = app
        .output_lines
        .iter()
        .skip(scroll)
        .take(visible_rows)
        .map(|(text, style)| Line::from(Span::styled(text.clone(), *style)))
        .collect();

    let paragraph = Paragraph::new(lines)
        .style(Style::default().bg(BG))
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, inner);
}

fn draw_dialog(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let title = match app.dialog {
        DialogMode::SaveAs => " Save As (Enter to confirm, Esc to cancel) ",
        DialogMode::LoadFrom => " Load File (Enter to confirm, Esc to cancel) ",
        DialogMode::None => return,
    };

    let width = 60.min(area.width.saturating_sub(4));
    let height = 3;
    let x = (area.width.saturating_sub(width)) / 2;
    let y = (area.height.saturating_sub(height)) / 2;

    let dialog_area = Rect::new(x, y, width, height);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(FG))
        .style(Style::default().bg(Color::Rgb(0, 20, 10)));

    let inner = block.inner(dialog_area);

    let input_text = format!("{}_", app.dialog_input);
    let paragraph = Paragraph::new(Line::from(Span::styled(
        input_text,
        Style::default().fg(FG),
    )))
    .style(Style::default().bg(Color::Rgb(0, 20, 10)));

    // Clear the area behind dialog
    let clear = Paragraph::new("")
        .style(Style::default().bg(Color::Rgb(0, 20, 10)));
    frame.render_widget(clear, dialog_area);
    frame.render_widget(block, dialog_area);
    frame.render_widget(paragraph, inner);
}

// ---------------------------------------------------------------------------
// Event handling
// ---------------------------------------------------------------------------
fn handle_event(app: &mut App, ev: Event) {
    match ev {
        Event::Key(key) => {
            if app.dialog != DialogMode::None {
                handle_dialog_key(app, key);
            } else {
                handle_editor_key(app, key);
            }
        }
        Event::Resize(_, _) => {} // terminal will redraw
        _ => {}
    }
}

fn handle_dialog_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc => {
            app.dialog = DialogMode::None;
            app.dialog_input.clear();
        }
        KeyCode::Enter => {
            let path = app.dialog_input.clone();
            if !path.is_empty() {
                match app.dialog {
                    DialogMode::SaveAs => app.save_file(&path),
                    DialogMode::LoadFrom => app.load_file(&path),
                    DialogMode::None => {}
                }
            }
            app.dialog = DialogMode::None;
            app.dialog_input.clear();
        }
        KeyCode::Backspace => {
            app.dialog_input.pop();
        }
        KeyCode::Char(c) => {
            app.dialog_input.push(c);
        }
        _ => {}
    }
}

fn handle_editor_key(app: &mut App, key: KeyEvent) {
    // Ctrl combinations
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('q') => {
                app.running = false;
                return;
            }
            _ => {}
        }
    }

    match key.code {
        // Function keys
        KeyCode::F(5) => {
            app.compile_shader();
        }
        KeyCode::F(6) => {
            app.compile_and_run();
        }
        KeyCode::F(2) => {
            app.dialog = DialogMode::SaveAs;
            app.dialog_input = app.file_path.clone();
        }
        KeyCode::F(3) => {
            app.dialog = DialogMode::LoadFrom;
            app.dialog_input = app.file_path.clone();
        }
        KeyCode::Tab => {
            app.cycle_template();
        }

        // Navigation
        KeyCode::Up => {
            if app.cursor_row > 0 {
                app.cursor_row -= 1;
            }
            app.clamp_cursor();
        }
        KeyCode::Down => {
            if app.cursor_row + 1 < app.lines.len() {
                app.cursor_row += 1;
            }
            app.clamp_cursor();
        }
        KeyCode::Left => {
            if app.cursor_col > 0 {
                app.cursor_col -= 1;
            } else if app.cursor_row > 0 {
                app.cursor_row -= 1;
                app.cursor_col = app.lines[app.cursor_row].chars().count();
            }
        }
        KeyCode::Right => {
            let char_count = app.lines[app.cursor_row].chars().count();
            if app.cursor_col < char_count {
                app.cursor_col += 1;
            } else if app.cursor_row + 1 < app.lines.len() {
                app.cursor_row += 1;
                app.cursor_col = 0;
            }
        }
        KeyCode::Home => {
            app.cursor_col = 0;
        }
        KeyCode::End => {
            app.cursor_col = app.lines[app.cursor_row].chars().count();
        }
        KeyCode::PageUp => {
            app.cursor_row = app.cursor_row.saturating_sub(20);
            app.clamp_cursor();
        }
        KeyCode::PageDown => {
            app.cursor_row = (app.cursor_row + 20).min(app.lines.len().saturating_sub(1));
            app.clamp_cursor();
        }

        // Editing
        KeyCode::Char(c) => {
            app.insert_char(c);
        }
        KeyCode::Enter => {
            app.insert_newline();
        }
        KeyCode::Backspace => {
            app.backspace();
        }
        KeyCode::Delete => {
            app.delete();
        }

        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Install panic hook to restore terminal on panic
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen);
        original_hook(info);
    }));

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut app = App::new();

    // Main loop
    while app.running {
        app.update_status();
        app.ensure_cursor_visible(
            terminal.size().map(|s| s.height as usize).unwrap_or(24).saturating_sub(4),
        );

        draw(&mut terminal, &app);

        // Poll for events with a small timeout for smooth UI
        if event::poll(Duration::from_millis(50))? {
            let ev = event::read()?;
            handle_event(&mut app, ev);
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    println!("gpu-shader-playground exited.");
    Ok(())
}
