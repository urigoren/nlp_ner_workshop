from flask import Flask, request, send_from_directory, render_template
import traceback
import sys
sys.path.append('python')
from style_predict import autotag, load_model
app = Flask(__name__)
model, params = load_model('model')


@app.route('/assets/<path:path>')
def serve_assets(path):
    return send_from_directory('assets', path)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('assets', 'favicon.ico')


@app.after_request
def add_no_cache(response):
    if request.endpoint != "static":
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Pragma"] = "no-cache"
    return response


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


@app.route('/style', methods=["GET", "POST"])
def style():
    try:
        txt = request.form["txt"]
        return autotag(txt, model, params)
    except Exception as e:
        return f"<h2>{e}</h2>"+traceback.format_exc().replace('\n', '<br>')


@app.route('/')
def root():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
