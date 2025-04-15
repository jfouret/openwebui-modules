# Using Tool Calls for Collapsible UI Elements in Open WebUI

This document explains how to use the tool_calls mechanism in Open WebUI to create collapsible UI elements that can display information about your tool's processing.

## Overview

Open WebUI provides a mechanism for tools to display information in collapsible UI elements using HTML `<details>` and `<summary>` tags. This is used for various purposes, including:

1. **Reasoning/Thinking**: Displaying the model's reasoning process
2. **Tool Calls**: Displaying tool execution and results
3. **Code Interpreter**: Displaying code execution and results

These collapsible elements provide a clean way to show detailed information without cluttering the main chat interface.

## Tool Calls Implementation

The tool_calls mechanism is specifically designed to display structured information about tool execution, primarily the arguments passed to the tool and the results returned by the tool. It is not intended for displaying detailed processing steps or custom HTML content.

Here's how it works:

1. A content block of type "tool_calls" is created with information about the tool call
2. The content block is serialized into HTML with `<details>` tags
3. The serialized content is sent to the frontend using the `__event_emitter__` function
4. When the tool execution is complete, the results are added to the content block
5. The content blocks are serialized again, this time with the results included

## HTML Structure

The HTML structure for tool_calls looks like this:

```html
<!-- For a completed tool call with results -->
<details type="tool_calls" done="true" id="tool-call-id" name="tool-name" arguments="{...}" result="{...}" files="{...}">
<summary>Tool Executed</summary>
</details>

<!-- For a tool call in progress -->
<details type="tool_calls" done="false" id="tool-call-id" name="tool-name" arguments="{...}">
<summary>Executing...</summary>
</details>
```

**Important Note**: The tool_calls mechanism is designed to display only the arguments and results (and eventually files) of a tool execution. It does not support custom HTML content inside the `<details>` element. The content is limited to a `<summary>` element with fixed text.

## Implementation in Your Tool

To implement this in your tool, you can use the `__event_emitter__` function that is passed to your tool as part of the `extra_params` dictionary. Here's an example:

```python
async def my_tool_function(**params):
    # Extract event_emitter from extra_params
    event_emitter = params.get("__event_emitter__")
    
    if event_emitter:
        # Send a status update to show that the tool is starting
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "description": "Starting my tool",
                    "done": False,
                },
            }
        )
        
        # Create a tool_call content block to display arguments
        # This will be displayed as a collapsible element in the UI
        await event_emitter(
            {
                "type": "message",
                "data": {
                    "content": f'<details type="tool_calls" done="false" id="my-tool-id" name="my-tool" arguments="{html.escape(json.dumps(params))}">\n<summary>Processing...</summary>\n</details>'
                },
            }
        )
        
        # Do your tool's processing...
        result = process_data(params)
        
        # Update the tool_call content block with the results
        await event_emitter(
            {
                "type": "replace",
                "data": {
                    "content": f'<details type="tool_calls" done="true" id="my-tool-id" name="my-tool" arguments="{html.escape(json.dumps(params))}" result="{html.escape(json.dumps(result))}">\n<summary>Tool Executed</summary>\n</details>'
                },
            }
        )
        
        # Send a status update to show that the tool is complete
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "description": "Tool completed",
                    "done": True,
                },
            }
        )
    
    # Return the result
    return result
```

## Using Regular Messages for Detailed Information

For displaying detailed processing steps or custom HTML content, it's recommended to use regular messages with HTML `<details>` elements. Here's an example:

```python
await event_emitter(
    {
        "type": "message",
        "data": {
            "content": """
<details>
<summary>Tool Processing Details</summary>
### Processing Steps
- Step 1: Processed input data
- Step 2: Performed calculation
- Step 3: Generated results
</details>
"""
        },
    }
)
```

This approach allows you to include rich, formatted content inside the `<details>` element, which will be displayed to the user when they expand the collapsible section.

## Advanced Usage

For more advanced usage, you can create custom collapsible elements with different types. Here are some examples:

### Custom Tool Information

```python
await event_emitter(
    {
        "type": "message",
        "data": {
            "content": """
<details>
<summary>Tool Information</summary>
<pre>
{detailed_info}
</pre>
</details>
"""
        },
    }
)
```

### Processing Steps

```python
await event_emitter(
    {
        "type": "message",
        "data": {
            "content": """
<details>
<summary>Processing Steps</summary>
<ol>
<li>Step 1: {step1_details}</li>
<li>Step 2: {step2_details}</li>
<li>Step 3: {step3_details}</li>
</ol>
</details>
"""
        },
    }
)
```

### Debug Information

```python
await event_emitter(
    {
        "type": "message",
        "data": {
            "content": """
<details>
<summary>Debug Information</summary>
<pre>
{debug_info}
</pre>
</details>
"""
        },
    }
)
```

## Conclusion

While the tool_calls mechanism in Open WebUI is designed specifically for displaying structured information about tool execution (arguments and results), you can use regular messages with HTML `<details>` elements to create rich, interactive UI elements that provide detailed information about your tool's processing. This can greatly enhance the user experience and make your tools more transparent and user-friendly.
