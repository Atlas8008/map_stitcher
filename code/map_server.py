import os

from time import sleep
from threading import Thread
from werkzeug.utils import secure_filename
from flask import Flask, render_template, send_from_directory, request, url_for, redirect

from map_processing import process_image, image_as_numpy, save_numpy_as_img


app = Flask(__name__)

upload_folder = "./web/temp/uploads"
processed_folder = "./web/temp/processed"

app.config["UPLOAD_FOLDER"] = upload_folder


def delayed_delete_file(fpath):
    sleep(60)
    os.remove(fpath)
    print("Removed file", fpath)


@app.route("/", methods=["GET", "POST"])
def main_page():
    if "file" not in request.files:
        return render_template("main.html")

    file = request.files["file"]

    if file.filename == "":
        return render_template("main.html", status="Invalid file.")

    if file:
        fname = secure_filename(file.filename)

        fpath = os.path.join(upload_folder, fname)
        processed_fpath = os.path.join(processed_folder, fname)

        file.save(fpath)
        print("Saved file", fpath)

        img_ary = image_as_numpy(fpath, max_height=int(request.form["height"]))
        processed_image = process_image(
            img_ary,
            rows=int(request.form["rows"]),
            columns=int(request.form["cols"]),
            regression_considered_points=int(request.form["n_reg_points"]),
        )
        save_numpy_as_img(processed_image, processed_fpath)

        # Delete uploaded image
        os.remove(fpath)
        print("Removed file", fpath)

        t = Thread(target=delayed_delete_file, args=[processed_fpath], daemon=True)
        t.start()

        return redirect(url_for("download_file", name=fname))

    return render_template("main.html")


@app.route("/download/<name>")
def download_file(name):
    return send_from_directory(processed_folder, name)


app.run(host="0.0.0.0")
