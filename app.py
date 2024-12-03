from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess

app = Flask(__name__)

# Path for the data directory to save files
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for the upload
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # After uploading the file, redirect to the analysis page
        return render_template('analysis.html', filename=file.filename)

    return 'File not allowed'

@app.route('/visualize', methods=["GET"])
def visualize():
    # Run the `main.py` file in parallel to trigger the analysis process
    print("[INFO] Running main.py...")

    # Run main.py asynchronously in the background
    subprocess.Popen(["python", "main.py"])

    # Redirect to the new page with visualization options
    return redirect(url_for("visualize_results"))

@app.route("/visualize-results")
def visualize_results():
    # Render the new visualization page with buttons for model analysis and prediction analysis
    return render_template("visualize.html")  # Create visualize.html for this

if __name__ == '__main__':
    app.run(debug=True)
