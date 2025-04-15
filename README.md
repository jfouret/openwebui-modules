# Open WebUI Modules

This repository contains modules for extending the capabilities of Open WebUI, including tools and functions.

## Project Structure

```
openwebui-modules/
├── docs/                  # Documentation files
│   └── features/
│       └── plugin/        # Documentation for plugins (tools and functions)
│           ├── tools/     # Documentation for tools development
│           └── functions/ # Documentation for functions development
├── functions/             # Functions to extend Open WebUI capabilities
│   ├── actions/           # Action functions
│   ├── filters/           # Filter functions
│   └── pipes/             # Pipe functions
└── tools/                 # Tools to extend LLM capabilities
    └── web_fetcher.py     # Tool for fetching web content with link following
```

## Documentation References

The most important documentation files for development are:

- `docs/features/plugin/index.mdx` - Overview of tools and functions
- `docs/features/plugin/tools/index.mdx` - General information about tools
- `docs/features/plugin/tools/development.mdx` - Detailed guide for developing tools
- `docs/features/plugin/functions/index.mdx` - General information about functions
- `docs/features/plugin/functions/action.mdx` - Guide for developing action functions
- `docs/features/plugin/functions/filter.mdx` - Guide for developing filter functions
- `docs/features/plugin/functions/pipe.mdx` - Guide for developing pipe functions

## Local Development Notes

Additional notes for developers working with Open WebUI:

- `notes_on_event_emitter.md` - Detailed information about the `__event_emitter__` and `__event_call__` functions
- `notes_on_tool_calls.md` - Guide for using tool calls to create collapsible UI elements

## Development Quick Reference

### Tools Development

Tools extend the abilities of LLMs, allowing them to collect real-world, real-time data. Each tool is a single Python file in the `tools/` directory with:

1. A top-level docstring containing metadata (title, author, description, etc.)
2. A `Tools` class with methods that implement the tool's functionality
3. Optional `Valves` and `UserValves` classes for configuration

For detailed information on developing tools, refer to `docs/features/plugin/tools/development.mdx`.

### Functions Development

Functions extend the capabilities of Open WebUI itself and are organized into:

1. **Actions**: Add new features to the UI (in `functions/actions/`)
2. **Filters**: Process or modify data (in `functions/filters/`)
3. **Pipes**: Transform data between different formats (in `functions/pipes/`)

For detailed information on developing functions, refer to the documentation in `docs/features/plugin/functions/`.

## Web Fetcher Tool

The repository includes a web fetcher tool (`tools/web_fetcher.py`) that demonstrates how to create a tool for Open WebUI. This tool:

- Fetches content from a given URL
- Parses HTML to extract readable content
- Converts HTML to markdown using the markdownify package
- Can identify and follow links on the same domain up to a specified depth
- Provides configuration options through UserValves
