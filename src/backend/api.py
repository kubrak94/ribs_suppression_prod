import argparse
import json
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, redirect, url_for, request, flash, send_from_directory
from werkzeug.utils import secure_filename

from src.model import Model

UPLOAD_FOLDER = './src/backend/static'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

original = []
processed = []


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'ribs_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['ribs_image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = str(Path(app.config['UPLOAD_FOLDER']).joinpath(filename))
            file.save(file_path)
            original.append(file.filename)
            processed_path = net.predict(file_path)
            processed.append(Path(processed_path).name)
            return redirect(url_for('show_results'))
    return render_template('fl.html')


@app.route('/results')
def show_results():
    a = {}
    a['original'] = original[0]
    a['processed'] = processed[0]
    original.pop()
    processed.pop()
    return render_template("imgs.html", args=a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, default='10.90.136.163', help="Host of API")
    parser.add_argument("-p", "--port", type=int, default=5005, help="Port on which to run API")
    parser.add_argument("-m", "--model", type=str, default="./models/fpn_final_model.pth.tar",
                        help="Path to model file")
    parser.add_argument("-s", "--save_images_path", type=str, default="./src/backend/static",
                        help="Path where images will be saved")
    args = parser.parse_args()

    net = Model(args.model, args.save_images_path)

    app.run(debug=True, host=args.ip, port=args.port, use_reloader=False)
