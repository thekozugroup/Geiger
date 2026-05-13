# Tools Reference

## file_search

Searches for files matching a specified pattern within a directory tree.

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `pattern` | string | Yes | Glob pattern to match files (e.g., `*.txt`, `**/*.js`) |
| `directory` | string | No | Root directory to search in. Defaults to current working directory |
| `recursive` | boolean | No | Whether to search subdirectories. Defaults to `true` |
| `max_results` | number | No | Maximum number of results to return. Defaults to `100` |

### Returns

```json
{
  "files": [
    {
      "path": "string",
      "name": "string",
      "size": "number",
      "modified": "string (ISO 8601)"
    }
  ],
  "total_count": "number"
}
```

### Example

```json
{
  "name": "file_search",
  "arguments": {
    "pattern": "*.md",
    "directory": "/Users/michaelwong/projects",
    "recursive": true
  }
}
```

---

## image_process

Applies transformations and processing operations to an image.

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `input_path` | string | Yes | Path to the input image file |
| `output_path` | string | Yes | Path where the processed image will be saved |
| `operation` | string | Yes | Operation to perform: `resize`, `crop`, `rotate`, `grayscale`, `blur`, `sharpen` |
| `params` | object | No | Operation-specific parameters |
| `format` | string | No | Output format override: `jpeg`, `png`, `webp` |

### Returns

```json
{
  "success": true,
  "output_path": "string",
  "dimensions": {
    "width": "number",
    "height": "number"
  },
  "file_size": "number"
}
```

### Example

```json
{
  "name": "image_process",
  "arguments": {
    "input_path": "/images/photo.jpg",
    "output_path": "/images/photo_thumb.png",
    "operation": "resize",
    "params": {
      "width": 200,
      "height": 200,
      "maintain_aspect": true
    }
  }
}
```

---

## data_query

Executes a SQL query against a database and returns the results.

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `connection_string` | string | Yes | Database connection string or alias |
| `query` | string | Yes | SQL query to execute |
| `parameters` | array | No | Query parameters for prepared statements |
| `timeout` | number | No | Query timeout in milliseconds. Defaults to `30000` |
| `max_rows` | number | No | Maximum rows to return. Defaults to `1000` |

### Returns

```json
{
  "rows": [
    {
      "column_name": "value"
    }
  ],
  "row_count": "number",
  "execution_time_ms": "number",
  "columns": ["string"]
}
```

### Example

```json
{
  "name": "data_query",
  "arguments": {
    "connection_string": "postgres://localhost/mydb",
    "query": "SELECT * FROM users WHERE status = $1 LIMIT $2",
    "parameters": ["active", 100],
    "timeout": 5000
  }
}
```

### Errors

| Code | Description |
|------|-------------|
| `CONNECTION_FAILED` | Unable to connect to the database |
| `QUERY_TIMEOUT` | Query exceeded the specified timeout |
| `INVALID_QUERY` | SQL syntax error or invalid query |
| `ROWS_EXCEEDED` | Result set exceeds max_rows limit |