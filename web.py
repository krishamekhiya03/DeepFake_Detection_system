from flask import Flask, render_template, request, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, FloatField, SubmitField
from wtforms.validators import InputRequired
import os
import cv2
import hashlib
import numpy as np
import io
import base64
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

class ImageForgeryDetectorForm(FlaskForm):
    original_image = FileField('Original Image', validators=[InputRequired()])
    tampered_image = FileField('Tampered Image', validators=[InputRequired()])
    ssim_threshold = FloatField('SSIM Threshold', default=0.9, validators=[InputRequired()])
    submit = SubmitField('Detect Forgery')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ImageForgeryDetectorForm()
    original_preview = None
    tampered_preview = None

    if form.validate_on_submit():
        original_path = save_uploaded_file(form.original_image)
        tampered_path = save_uploaded_file(form.tampered_image)
        ssim_threshold = form.ssim_threshold.data

        result, histograms, forged_area_img = detect_image_forgery(original_path, tampered_path, ssim_threshold)

        # Image Previews
        original_preview = encode_image_preview(original_path)
        tampered_preview = encode_image_preview(tampered_path)

        return render_template('result.html', result=result, histograms=histograms,
                               original_preview=original_preview, tampered_preview=tampered_preview,
                               forged_area_img=forged_area_img)

    return render_template('index.html', form=form, original_preview=original_preview, tampered_preview=tampered_preview)

def save_uploaded_file(file_field):
    if file_field.data:
        filename = file_field.data.filename
        file_path = os.path.join('uploads', filename)
        file_field.data.save(file_path)
        return file_path
    return None

def calculate_hashes(image_path):
    with open(image_path, 'rb') as f:
        content = f.read()
        md5_hash = hashlib.md5(content).hexdigest()
        sha256_hash = hashlib.sha256(content).hexdigest()
    return md5_hash, sha256_hash

def image_similarity(img1, img2):
    ssim = cv2.compare_ssim(img1, img2)
    return ssim

def detect_image_forgery(original_path, tampered_path, ssim_threshold):
    original_img = cv2.imread(original_path)
    tampered_img = cv2.imread(tampered_path)

    original_img = cv2.resize(original_img, (256, 256))
    tampered_img = cv2.resize(tampered_img, (256, 256))

    original_md5, original_sha256 = calculate_hashes(original_path)
    tampered_md5, tampered_sha256 = calculate_hashes(tampered_path)

    if original_md5 != tampered_md5:
        result = "MD5 Hash Mismatch: The images are forged."
    else:
        result = "MD5 Hash Match: The images are similar."
        similarity = np.sum(np.abs(original_img - tampered_img)) / (original_img.shape[0] * original_img.shape[1] * original_img.shape[2])

        if similarity < ssim_threshold:
            result += f"\nImage similarity score: {similarity}\nPossible forgery detected."
        else:
            result += f"\nImage similarity score: {similarity}\nImages are likely similar."

    result += f"\nOriginal Image MD5: {original_md5}\nOriginal Image SHA-256: {original_sha256}"
    result += f"\nTampered Image MD5: {tampered_md5}\nTampered Image SHA-256: {tampered_sha256}"

    forged_area_img = highlight_forged_area(original_img, tampered_img)

    histograms = plot_histograms(original_img, tampered_img)

    return result, histograms, forged_area_img

def highlight_forged_area(original_img, tampered_img):
    difference_img = cv2.absdiff(original_img, tampered_img)
    _, thresh = cv2.threshold(cv2.cvtColor(difference_img, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tampered_with_forgery = tampered_img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(tampered_with_forgery, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Highlight in red

    _, forged_area_buffer = cv2.imencode('.png', tampered_with_forgery)
    forged_area_img = base64.b64encode(forged_area_buffer).decode('utf-8')
    return forged_area_img

def plot_histograms(original_img, tampered_img):
    original_hist = calculate_histogram(original_img)
    tampered_hist = calculate_histogram(tampered_img)

    return original_hist, tampered_hist

def calculate_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(cdf_normalized, color='blue')
    ax.hist(gray.flatten(), 256, [0, 256], color='blue')
    ax.set_title('Image Histogram')

    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    return output.getvalue()

def encode_image_preview(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/download_result', methods=['POST'])
def download_result():
    result_text = request.form['result']
    file_content = f"Image Forgery Detection Result: We are providing this result based on MD5 algorithm and SHA algorithm,with this algorithm one can identify if there is forgery in their image or not !! \n\n{result_text}"
    file = io.BytesIO(file_content.encode())
    return send_file(file, as_attachment=True, download_name='forgery_detection_result.txt')

if __name__ == '__main__':
    app.run(debug=True)

