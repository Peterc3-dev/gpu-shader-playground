# gpu-shader-playground

GLSL compute shader editor with syntax highlighting and live Vulkan execution in a split-pane TUI.

## Features

- Built-in text editor with GLSL syntax highlighting (keywords, types, builtins, numbers, comments, directives)
- Compile shaders via `glslangValidator` with inline error/warning display
- Execute compiled SPIR-V on the GPU via Vulkan and view output buffer values
- GPU timestamp queries for kernel timing
- 4 starter templates: SAXPY, parallel reduction, image processing (box blur), Mandelbrot
- Cycle through templates with Tab
- Save/load shader files with dialog prompts
- Auto-indent on newline, cursor tracking, scrollable editor
- Phosphor-green color scheme

## Install

```
cargo build --release
```

Requires `glslangValidator` (from the Vulkan SDK or `glslang-tools` package) and working Vulkan drivers.

## Usage

```
gpu-shader-playground
```

Opens with the SAXPY template loaded. Edit the shader, press F5 to compile, F6 to compile and run.

## Keybindings

| Key       | Action                          |
|-----------|---------------------------------|
| `F5`      | Compile shader                  |
| `F6`      | Compile and execute on GPU      |
| `F2`      | Save file                       |
| `F3`      | Load file                       |
| `Tab`     | Cycle shader template           |
| Arrows    | Move cursor                     |
| `Home`    | Jump to line start              |
| `End`     | Jump to line end                |
| `PgUp/Dn` | Scroll by 20 lines            |
| `Ctrl+Q`  | Quit                            |

---

Built with Rust + ratatui + ash.
