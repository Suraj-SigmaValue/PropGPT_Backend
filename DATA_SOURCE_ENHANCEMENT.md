# Data Source Display Enhancement

## Overview
This enhancement adds data source information to the API response, allowing the frontend to display which data source (RERA, IGR, IGR-CGDB, DA, etc.) is used for each column/metric.

## Changes Made

### 1. Created `core/source_utils.py`
A new utility module that provides functions to enrich columns with source information:

- **`get_columns_with_sources(selected_columns, comparison_type)`**: Main function that maps each column to its data source
- **`format_data_sources_text(columns_with_sources)`**: Helper function to format the data as readable text

### 2. Updated `api/views.py`
Modified the `MainQueryView` to include source information in the response:

```python
# Get selected columns with source information
from core.source_utils import get_columns_with_sources
selected_columns = final_state.get('selected_columns', [])
columns_with_sources = get_columns_with_sources(selected_columns, comparison_type)

# Added to response
response_data = {
    ...
    'columns_with_sources': columns_with_sources,  # New field
}
```

### 3. Updated `api/serializers.py`
Added the `columns_with_sources` field to `QueryResponseSerializer`:

```python
columns_with_sources = serializers.ListField(
    child=serializers.DictField(),
    required=False
)  # Added: columns enriched with source information
```

## API Response Format

The API now returns an additional field `columns_with_sources` in the response.

**IMPORTANT:** Column names are displayed in their **original format** from `COLUMN_MAPPING` (e.g., `"flat_sold - igr"` with hyphens), NOT the normalized format (e.g., `"flat_sold igr"`).

```json
{
  "response_text": "...",
  "mapping_keys": [...],
  "selected_columns": [...],
  "columns_with_sources": [
    {
      "column": "flat - age range wise total sales",
      "source": "IGR-CGDB"
    },
    {
      "column": "office - age range wise total sales",
      "source": "RERA"
    },
    {
      "column": "shop - age range wise total sales",
      "source": "IGR-CGDB"
    }
  ],
  "input_tokens": 1234,
  "output_tokens": 567,
  "cached": false
}
```

## Frontend Integration

### Option 1: Display as a List

```javascript
// In your frontend component
const displayDataSources = (columnsWithSources) => {
  return (
    <div className="data-sources-section">
      <h3>Data Sources</h3>
      <ul>
        {columnsWithSources.map((item, index) => (
          <li key={index}>
            <strong>{item.column}</strong>: {item.source}
          </li>
        ))}
      </ul>
    </div>
  );
};
```

### Option 2: Display as a Table

```javascript
const displayDataSourcesTable = (columnsWithSources) => {
  return (
    <div className="data-sources-section">
      <h3>Data Sources</h3>
      <table>
        <thead>
          <tr>
            <th>Column Name</th>
            <th>Data Source</th>
          </tr>
        </thead>
        <tbody>
          {columnsWithSources.map((item, index) => (
            <tr key={index}>
              <td>{item.column}</td>
              <td><span className="source-badge">{item.source}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
```

### Option 3: Display as Formatted Text

```javascript
const displayDataSourcesText = (columnsWithSources) => {
  const text = columnsWithSources
    .map(item => `${item.column}: ${item.source}`)
    .join('\n');
  
  return (
    <div className="data-sources-section">
      <h3>Data Sources</h3>
      <pre>{text}</pre>
    </div>
  );
};
```

## Example Usage in React Component

```jsx
import React from 'react';

const QueryResponse = ({ responseData }) => {
  const { response_text, columns_with_sources } = responseData;
  
  return (
    <div className="query-response">
      {/* Main response */}
      <div className="response-text">
        <ReactMarkdown>{response_text}</ReactMarkdown>
      </div>
      
      {/* Data sources section */}
      {columns_with_sources && columns_with_sources.length > 0 && (
        <div className="data-sources">
          <h3>Data Sources</h3>
          <div className="sources-grid">
            {columns_with_sources.map((item, index) => (
              <div key={index} className="source-item">
                <span className="column-name">{item.column}</span>
                <span className="source-badge">{item.source}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default QueryResponse;
```

## CSS Styling Example

```css
.data-sources {
  margin-top: 20px;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 8px;
}

.data-sources h3 {
  margin-top: 0;
  color: #2c3e50;
  font-size: 16px;
  margin-bottom: 12px;
}

.sources-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 10px;
}

.source-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background-color: white;
  border-radius: 4px;
  border-left: 3px solid #3498db;
}

.column-name {
  font-size: 13px;
  color: #34495e;
  flex: 1;
}

.source-badge {
  padding: 4px 8px;
  background-color: #3498db;
  color: white;
  border-radius: 4px;
  font-size: 11px;
  font-weight: bold;
}
```

## Data Source Mapping

The system uses three SOURCE_MAPPING dictionaries in `core/mapping.py`:

1. **SOURCE_MAPPING_Location**: For location-wise analysis
2. **SOURCE_MAPPING_City**: For city-wise analysis
3. **SOURCE_MAPPING_Project**: For project-wise analysis

### Source Types:
- **RERA**: Real Estate Regulatory Authority data
- **IGR**: Inspector General of Registration data
- **IGR-CGDB**: Combined IGR and CGDB (Central Government Database) data
- **DA**: Development Agreement data
- **IGR+RERA**: Mixed sources
- **Unknown**: Source not found (fallback)

## Testing

To test the functionality:

1. Start the Django server:
   ```bash
   python manage.py runserver
   ```

2. Send a query request to `/api/query/`

3. Check the response for the `columns_with_sources` field

4. Example response structure:
   ```json
   {
     "columns_with_sources": [
       {"column": "flat - age range wise total sales", "source": "IGR-CGDB"},
       {"column": "office - age range wise total sales", "source": "RERA"},
       {"column": "others - age range wise total sales", "source": "IGR-CGDB"}
     ]
   }
   ```

## Backward Compatibility

This enhancement maintains full backward compatibility:
- The `columns_with_sources` field is marked as `required=False` in the serializer
- Existing API clients that don't expect this field will continue to work
- Frontend can check for the presence of this field before displaying

## Benefits

1. **Transparency**: Users can see exactly which data source is being used for each metric
2. **Trust**: Builds confidence in the data by showing authoritative sources
3. **Debugging**: Helps identify data quality issues by source
4. **Compliance**: Meets regulatory requirements for data source disclosure

## Troubleshooting

### Issue: All columns showing "Unknown" source

**Symptom**: All columns in `columns_with_sources` show `"Unknown"` as the source instead of the correct source (RERA, IGR, etc.)

**Root Cause**: The column names in `COLUMN_MAPPING` (from `mapping.py`) have different formatting than the normalized column names used in the system. For example:
- In `COLUMN_MAPPING`: `"flat_sold - igr"` (with hyphens and spaces)
- In selected_columns: `"flat_sold igr"` (normalized, hyphens replaced with spaces)

**Fix Applied**: The `get_columns_with_sources()` function now uses the `normalize_colname()` function to normalize both:
1. Column names from `COLUMN_MAPPING` when building the reverse mapping
2. Selected column names when looking them up

This ensures consistent comparison using the normalized format:
```python
from .core_utils import normalize_colname

# Normalize columns from COLUMN_MAPPING
normalized_col = normalize_colname(str(col))  # "flat_sold - igr" -> "flat_sold igr"
column_to_key[normalized_col] = mapping_key

# Normalize selected columns
column_normalized = normalize_colname(str(column))
mapping_key = column_to_key.get(column_normalized)
```

**Test**: Run `python test_source_mapping.py` to verify all columns map correctly.

## Notes

- The source information is derived from the mapping keys, not individual columns
- Each mapping key is associated with a source in the SOURCE_MAPPING dictionaries
- If a column doesn't match any mapping key, it will show "Unknown" as the source
