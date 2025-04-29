import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from app.chunking_test import process_single_pdf

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SUMMARY_FOLDER = os.path.join(os.getcwd(), 'summaries')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SUMMARY_FOLDER'] = SUMMARY_FOLDER

    # ----- Routes -----
    @app.route('/')
    def home():
        pdfs = os.listdir(app.config['UPLOAD_FOLDER'])
        return render_template('home.html', pdfs=pdfs)

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            files = request.files.getlist('pdfs')
            filepaths = []

            for file in files:
                if file and file.filename.endswith('.pdf'):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    filepaths.append(filepath)

            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(generate_summary, filepaths)

            return redirect(url_for('home'))

        return render_template('upload.html')

    @app.route('/view/<filename>')
    def view_pdf(filename):
        summary_filename = filename.replace('.pdf', '.txt')
        summary_path = os.path.join(app.config['SUMMARY_FOLDER'], summary_filename)
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_text = f.read()
        else:
            summary_text = "Summary not available."
        return render_template('view.html', filename=filename, summary_text=summary_text)

    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/download_summary/<filename>')
    def download_summary(filename):
        return send_from_directory(app.config['SUMMARY_FOLDER'], filename, as_attachment=True)

    return app


def generate_summary(file_path):
    summary_text = process_single_pdf(file_path)
    pdf_filename = os.path.basename(file_path)
    summary_filename = pdf_filename.replace('.pdf', '.txt')
    summary_path = os.path.join(SUMMARY_FOLDER, summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    return summary_path


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
