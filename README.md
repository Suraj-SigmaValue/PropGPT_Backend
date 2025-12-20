# PropGPT Backend - Django REST API

## Setup Instructions

### 1. Create Virtual Environment
```bash
cd propgpt_backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy .env.example to .env
copy .env.example .env

# Edit .env and add your API keys
# Minimum required: OPENAI_API_KEY or GOOGLE_API_KEY
```

### 4. Copy Data Files
```bash
# Copy Excel file to data directory
# Ensure Pune_Grand_Summary.xlsx is in propgpt_backend/data/
```

### 5. Run Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Create Superuser (Optional)
```bash
python manage.py createsuperuser
```

### 7. Run Development Server
```bash
python manage.py runserver
```

API will be available at: `http://localhost:8000/api/`

---

## API Endpoints

### Main Endpoint
- `POST /api/query/` - Main query processing

### Data Management
- `POST /api/mappings/load/` - Load mappings for comparison type
- `POST /api/items/` - Get comparison items
- `GET /api/projects/recommendations/` - Get project recommendations

### Agents
- `POST /api/agents/planner/` - Planner agent (mapping keys)
- `POST /api/agents/column/` - Column selection agent
- `POST /api/agents/correction/` - Correction agent (HITL)

### LangGraph
- `POST /api/graph/execute/` - Execute graph workflow

### Cache
- `GET /api/cache/stats/` - Get cache statistics
- `POST /api/cache/clear/` - Clear cache

### Utilities
- `POST /api/relevance/` - Check query relevance
- `POST /api/feedback/` - Submit feedback (thumbs up/down)

---

## Project Structure

```
propgpt_backend/
├── manage.py
├── requirements.txt
├── .env.example
├── propgpt/          # Django project settings
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── core/             # Core logic modules (UNCHANGED from original)
│   ├── agents.py
│   ├── prompts.py
│   ├── config.py
│   ├── mapping.py
│   ├── response_cache.py
│   ├── graph_agent.py
│   └── core_utils.py
├── api/              # Django REST API layer
│   ├── models.py
│   ├── serializers.py
│   ├── views.py
│   ├── urls.py
│   └── admin.py
└── data/             # Data files
    └── Pune_Grand_Summary.xlsx
```

---

## Testing API

### Using curl:
```bash
# Get comparison items
curl -X POST http://localhost:8000/api/items/ \
  -H "Content-Type: application/json" \
  -d '{"comparison_type": "Location"}'

# Main query
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the supply analysis?",
    "items": ["pune"],
    "categories": ["Supply"],
    "comparison_type": "Location"
  }'
```

### Using Django Browsable API:
Visit `http://localhost:8000/api/` in your browser for interactive API testing.

---

## Notes

- **Zero Logic Modification**: All core functions are imported and called exactly as-is
- **Pickle Generation**: The .pkl file will be generated automatically on first run from the Excel file
- **Cache**: Response cache and vector cache directories will be created automatically
- **API Keys**: At minimum, you need either `OPENAI_API_KEY` or `GOOGLE_API_KEY`
