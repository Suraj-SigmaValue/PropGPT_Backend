import logging
from typing import List, Dict, Any
from datetime import datetime
import markdown
from django.template import Template, Context
from xhtml2pdf import pisa
from core.core_utils import count_tokens

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates structured, institutional-grade real estate reports.
    """
    
    SECTION_METADATA = {
        'executive_summary': {'title': 'Executive Summary / Key Insights', 'priority': 1, 'mandatory': True},
        'market_overview': {'title': 'Market Overview', 'priority': 2},
        'price_trend': {'title': 'Price & Trend Analysis', 'priority': 3},
        'location_snapshot': {'title': 'Location / Micromarket Snapshot', 'priority': 4},
        'comparable_transactions': {'title': 'Comparable Transactions', 'priority': 5},
        'demand_supply': {'title': 'Demand–Supply Indicators', 'priority': 6},
        'risk_assumptions': {'title': 'Risk & Assumptions', 'priority': 7},
        'data_sources': {'title': 'Data Sources & Methodology', 'priority': 8},
        'charts_visuals': {'title': 'Charts & Visuals', 'priority': 9},
        'valuation': {'title': 'Valuation / Price Band', 'priority': 10},
    }

    PRESETS = {
        'quick': ['executive_summary', 'market_overview', 'charts_visuals'],
        'institutional': list(SECTION_METADATA.keys())
    }

    def __init__(self, llm, context_text: str, chat_history: List[Dict[str, str]], query: str):
        self.llm = llm
        self.context_text = context_text
        self.chat_history = chat_history
        self.query = query
        self.sections_content = {}

    def generate_report(self, selected_sections: List[str] = None, preset: str = None) -> str:
        """
        Main entry point to generate the report HTML.
        """
        if preset and preset in self.PRESETS:
            selected_sections = self.PRESETS[preset]
        elif not selected_sections:
            selected_sections = ['executive_summary']

        # Ensure executive summary is always included
        if 'executive_summary' not in selected_sections:
            selected_sections.insert(0, 'executive_summary')

        # Generate each section
        for section in selected_sections:
            if section in self.SECTION_METADATA:
                content = self._generate_section_content(section)
                if content and content.strip():
                    self.sections_content[section] = content

        if not self.sections_content:
            self.sections_content['status'] = "No specific data could be analyzed for this report. The report is based on the available conversation history."

        return self._render_to_html()

    def _generate_section_content(self, section: str) -> str:
        """
        Calls LLM to generate content for a specific section based on context.
        """
        section_title = self.SECTION_METADATA[section]['title']
        
        # Build prompt for section generation
        prompt = f"""
        You are a senior real estate research analyst. Generate the '{section_title}' section for an institutional-grade report.
        
        USER QUERY: {self.query}
        
        CONTEXT DATA:
        {self.context_text[:8000] if self.context_text else "No retrieval data available."} 
        
        CHAT HISTORY SUMMARY:
        {self._get_chat_summary()}
        
        INSTRUCTIONS:
        1. Base your answer on the provided Context Data AND/OR Chat History.
        2. Use a professional, defensible, and structured tone.
        3. If it's the 'Executive Summary', provide 4-6 crisp bullet points.
        4. For other sections, use clear headings, sub-bullet points, and structured paragraphs.
        5. If there is NO DATA available for this specific section in the context or history, return exactly: "NO_DATA_AVAILABLE".
        6. Do NOT hallucinate.
        
        SECTION TO GENERATE: {section_title}
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, 'content', str(response))
            
            if "NO_DATA_AVAILABLE" in content:
                return None
                
            return content
        except Exception as e:
            logger.error(f"Error generating section {section}: {e}")
            return None

    def _get_chat_summary(self) -> str:
        summary = ""
        for msg in self.chat_history[-10:]: # Increased to Last 10 messages for better context
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            summary += f"{role.upper()}: {content[:500]}...\n"
        return summary

    def _render_to_html(self) -> str:
        """
        Renders the generated sections to a beautiful HTML template.
        """
        processed_sections = []
        # Define a simplified default if metadata is missing (safe fallback)
        default_meta = {'title': 'Report Section', 'priority': 99}
        
        for section, content in self.sections_content.items():
            # Replace the black square with proper rupee symbol if present
            content = content.replace('■', '₹')
            html_content = markdown.markdown(content, extensions=['extra', 'sane_lists'])
            
            meta = self.SECTION_METADATA.get(section, default_meta)
            if section == 'status':
                 meta = {'title': 'Report Status', 'priority': 0}

            processed_sections.append({
                'id': section,
                'title': meta['title'],
                'content': html_content
            })
        
        # Sort sections by priority (handle custom/fallback sections)
        # Note: We need to access priority from metadata again or store it in processed_sections
        # For simplicity, we trust the order or just render as is, but let's be safe.
        
        html_template = """
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page { 
                    size: A4; 
                    margin: 2.5cm 1.5cm;
                }
                @font-face {
                    font-family: 'Helvetica';
                    src: url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari&display=swap');
                }
                body { 
                    font-family: 'Helvetica', 'Arial', sans-serif; 
                    font-size: 11px; 
                    line-height: 1.6; 
                    color: #2c3e50; 
                }
                
                /* Ensure rupee symbol displays properly */
                .rupee {
                    font-family: 'Noto+Sans+Devanagari', sans-serif;
                }
                
                /* Institutional Branding */
                .header { border-bottom: 3px solid #1a252f; padding-bottom: 20px; margin-bottom: 30px; }
                .logo-text { font-size: 24px; font-weight: bold; color: #1a252f; }
                .report-title { font-size: 18px; color: #34495e; margin-top: 5px; }
                
                .meta { margin-bottom: 40px; font-size: 10px; color: #7f8c8d; }
                
                /* Sections */
                .section { margin-bottom: 35px; page-break-inside: avoid; }
                .section-title { 
                    background-color: #f8f9fa; 
                    border-left: 5px solid #2980b9; 
                    padding: 8px 15px; 
                    font-size: 14px; 
                    font-weight: bold; 
                    color: #2c3e50;
                    margin-bottom: 15px;
                    text-transform: uppercase;
                }
                
                /* Content Styling */
                .content { text-align: justify; }
                h3 { color: #2980b9; font-size: 12px; margin-top: 15px; border-bottom: 1px solid #ecf0f1; padding-bottom: 3px; }
                ul, ol { margin-left: 20px; }
                li { margin-bottom: 5px; }
                
                /* Tables */
                table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 10px; }
                th { background-color: #34495e; color: white; padding: 8px; text-align: left; }
                td { border: 1px solid #dee2e6; padding: 8px; }
                tr:nth-child(even) { background-color: #f8f9fa; }

                .footer { margin-top: 50px; font-size: 9px; color: #bdc3c7; text-align: center; border-top: 1px solid #eee; padding-top: 10px; }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="logo-text">PropGPT</div>
                <div class="report-title">Market Intelligence & Analytics Report</div>
            </div>
            
            <div class="meta">
                <strong>Project/Location:</strong> {{ query }} <br>
                <strong>Date:</strong> {{ date }} <br>
                <strong>Ref ID:</strong> {{ session_id }}
            </div>

            {% for section in sections %}
                <div class="section" id="{{ section.id }}">
                    <div class="section-title">{{ section.title }}</div>
                    <div class="content">{{ section.content|safe }}</div>
                </div>
            {% endfor %}

            <div class="footer">
                This report is generated by PropGPT AI Intelligence. The information provided is based on available market data and should be used for informational purposes only.
            </div>
        </body>
        </html>
        """
        
        return html_template, processed_sections

    def create_pdf(self, html_template: str, sections_data: list, session_id: str) -> bytes:
        """
        Converts HTML to PDF using pisa.
        """
        from io import BytesIO
        
        template = Template(html_template)
        context = Context({
            'sections': sections_data,
            'date': datetime.now().strftime("%d %B %Y"),
            'query': self.query,
            'session_id': session_id
        })
        rendered_html = template.render(context)
        
        result = BytesIO()
        # Explicitly set encoding to UTF-8
        pdf = pisa.CreatePDF(rendered_html.encode('utf-8'), dest=result, encoding='utf-8')
        
        if pdf.err:
            logger.error(f"PDF Generation Error: {pdf.err}")
            # Instead of returning None (which causes 500 error and corrupt download),
            # Return a simple error PDF
            error_pdf = BytesIO()
            pisa.CreatePDF(f"<html><body><h1>Report Generation Error</h1><p>An error occurred while generating the PDF structure.</p></body></html>", dest=error_pdf)
            return error_pdf.getvalue()
            
        return result.getvalue()
