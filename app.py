from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/input", methods=["POST", "GET"])
def input():
    if request.method == "POST":
        user_input = request.form["num"]
        return f"{int(user_input) * 2}"
    else:   
        return render_template("input_test.html")