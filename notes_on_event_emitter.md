# Event Emitter and Event Call in Open WebUI

This document provides detailed information about the `__event_emitter__` and `__event_call__` functions in Open WebUI, which are used for real-time communication between the backend and frontend.

## Overview

The `__event_emitter__` and `__event_call__` functions are part of the WebSocket communication system in Open WebUI. They allow tools and functions to send events and receive responses from the frontend in real-time.

- `__event_emitter__`: Sends events from the backend to the frontend
- `__event_call__`: Makes a call to the frontend and waits for a response

These functions are passed to tools and functions as part of the `extra_params` dictionary, allowing them to communicate with the frontend during execution.

## Implementation

The implementation of these functions is in `backend/open_webui/socket/main.py`:

### Event Emitter

```python
def get_event_emitter(request_info, update_db=True):
    async def __event_emitter__(event_data):
        user_id = request_info["user_id"]

        session_ids = list(
            set(
                USER_POOL.get(user_id, [])
                + (
                    [request_info.get("session_id")]
                    if request_info.get("session_id")
                    else []
                )
            )
        )

        for session_id in session_ids:
            await sio.emit(
                "chat-events",
                {
                    "chat_id": request_info.get("chat_id", None),
                    "message_id": request_info.get("message_id", None),
                    "data": event_data,
                },
                to=session_id,
            )

        if update_db:
            if "type" in event_data and event_data["type"] == "status":
                Chats.add_message_status_to_chat_by_id_and_message_id(
                    request_info["chat_id"],
                    request_info["message_id"],
                    event_data.get("data", {}),
                )

            if "type" in event_data and event_data["type"] == "message":
                message = Chats.get_message_by_id_and_message_id(
                    request_info["chat_id"],
                    request_info["message_id"],
                )

                content = message.get("content", "")
                content += event_data.get("data", {}).get("content", "")

                Chats.upsert_message_to_chat_by_id_and_message_id(
                    request_info["chat_id"],
                    request_info["message_id"],
                    {
                        "content": content,
                    },
                )

            if "type" in event_data and event_data["type"] == "replace":
                content = event_data.get("data", {}).get("content", "")

                Chats.upsert_message_to_chat_by_id_and_message_id(
                    request_info["chat_id"],
                    request_info["message_id"],
                    {
                        "content": content,
                    },
                )

    return __event_emitter__
```

### Event Call

```python
def get_event_call(request_info):
    async def __event_caller__(event_data):
        response = await sio.call(
            "chat-events",
            {
                "chat_id": request_info.get("chat_id", None),
                "message_id": request_info.get("message_id", None),
                "data": event_data,
            },
            to=request_info["session_id"],
        )
        return response

    return __event_caller__
```

## Message Types

The event system uses a variety of message types to communicate different kinds of information. Here are the main message types used in Open WebUI:

### Status Messages

Status messages are used to update the frontend about the progress of an operation:

```python
await event_emitter(
    {
        "type": "status",
        "data": {
            "action": "web_search",
            "description": "Generating search query",
            "done": False,
        },
    }
)
```

### Content Messages

Content messages are used to send content to be displayed in the chat:

```python
await event_emitter(
    {
        "type": "message",
        "data": {"content": "Some content to display"},
    }
)
```

### Replace Messages

Replace messages are used to replace the entire content of a message:

```python
await event_emitter(
    {
        "type": "replace",
        "data": {"content": "New content to replace the old content"},
    }
)
```

### Chat Completion Messages

Chat completion messages are used to update the frontend about the progress of a chat completion:

```python
await event_emitter(
    {
        "type": "chat:completion",
        "data": {
            "content": "Content of the completion",
            "done": True,
        },
    }
)
```

### Tool Execution Messages

Tool execution messages are used to request the frontend to execute a tool:

```python
tool_result = await event_caller(
    {
        "type": "execute:tool",
        "data": {
            "id": str(uuid4()),
            "name": tool_function_name,
            "params": tool_function_params,
            "server": tool.get("server", {}),
            "session_id": metadata.get("session_id", None),
        },
    }
)
```

### Python Execution Messages

Python execution messages are used to request the frontend to execute Python code:

```python
output = await event_caller(
    {
        "type": "execute:python",
        "data": {
            "id": str(uuid4()),
            "code": code,
            "session_id": metadata.get("session_id", None),
        },
    }
)
```

### Chat Request Messages

Chat request messages are used to request a chat completion:

```python
res = await event_caller(
    {
        "type": "request:chat:completion",
        "data": {
            "form_data": form_data,
            "model": models[form_data["model"]],
            "channel": channel,
            "session_id": session_id,
        },
    }
)
```

### Other Message Types

There are also other message types used for specific purposes:

- `chat:title`: Update the title of a chat
- `chat:tags`: Update the tags of a chat
- `message:reply`: Reply to a message
- `message:update`: Update a message
- `message:reaction:add`: Add a reaction to a message
- `message:reaction:remove`: Remove a reaction from a message
- `message:delete`: Delete a message

## Usage in Tools

When developing tools for Open WebUI, you can use the `__event_emitter__` and `__event_call__` functions to communicate with the frontend. These functions are passed to your tool as part of the `extra_params` dictionary:

```python
async def my_tool_function(**params):
    # Extract event_emitter and event_call from extra_params
    event_emitter = extra_params["__event_emitter__"]
    event_call = extra_params["__event_call__"]
    
    # Send a status update to the frontend
    await event_emitter(
        {
            "type": "status",
            "data": {
                "description": "Starting my tool",
                "done": False,
            },
        }
    )
    
    # Do some work...
    
    # Send a completion status to the frontend
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
    return "Tool result"
```

## Advanced Usage

For more advanced usage, you can use the `__event_call__` function to request the frontend to perform an action and wait for the result:

```python
async def my_advanced_tool_function(**params):
    # Extract event_call from extra_params
    event_call = extra_params["__event_call__"]
    
    # Request the frontend to execute a tool
    tool_result = await event_call(
        {
            "type": "execute:tool",
            "data": {
                "id": str(uuid4()),
                "name": "some_tool",
                "params": {"param1": "value1"},
                "session_id": extra_params["__metadata__"].get("session_id", None),
            },
        }
    )
    
    # Process the result
    processed_result = process_result(tool_result)
    
    # Return the processed result
    return processed_result
```

## Conclusion

The `__event_emitter__` and `__event_call__` functions are powerful tools for creating interactive and real-time experiences in Open WebUI. By understanding how to use these functions, you can create tools and functions that provide rich feedback and interactivity to users.
