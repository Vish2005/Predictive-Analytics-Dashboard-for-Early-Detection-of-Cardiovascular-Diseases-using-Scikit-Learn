import matplotlib.pyplot as plt
import io
from fpdf import FPDF
import base64

def get_health_tips(risk_level):
    return [
        "Maintain a healthy diet.",
        "Exercise regularly.",
        "Keep cholesterol levels under control.",
        "Visit a doctor if your risk level is high."
    ]

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Heart Disease Risk Assessment Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(patient_data, final_risk, model_predictions):
    pdf = PDFReport()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Final Assessment: {final_risk}', 0, 1, 'L')
    pdf.ln(5)
    
    # Patient Data
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Patient Profile:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    for key, value in patient_data.items():
        pdf.cell(0, 8, f"{key}: {value}", 0, 1, 'L')
    
    pdf.ln(5)
    
    # Model Predictions
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Model Breakdown:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    for model_name, pred in model_predictions.items():
        pdf.cell(0, 8, f"{model_name}: {pred}", 0, 1, 'L')

    pdf.ln(5)
    
    # Health Tips
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Recommended Health Tips:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    tips = get_health_tips(final_risk)
    for tip in tips:
        pdf.cell(0, 8, f"- {tip}", 0, 1, 'L')
        
    return pdf.output(dest='S').encode('latin1')
