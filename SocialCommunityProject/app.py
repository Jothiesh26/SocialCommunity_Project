from flask import request, redirect
from flask import Flask, render_template
from community_detection import run_analysis
from flask import send_file
from flask import request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run")
def run():
    results = run_analysis()
    return render_template("index.html", results=results)

@app.route('/download')
def download():
    return send_file("community_output.csv", as_attachment=True)

@app.route('/predict', methods=['POST'])
def predict():
    degree = float(request.form['degree'])
    clustering = float(request.form['clustering'])
    betweenness = float(request.form['betweenness'])
    closeness = float(request.form['closeness'])
    pagerank = float(request.form['pagerank'])

    features = [[degree, clustering, betweenness, closeness, pagerank]]

    prediction = model.predict(features)[0]

    return render_template("index.html", prediction=prediction)

@app.route("/add", methods=["POST"])
def add_data():
    import mysql.connector

    person1 = request.form["person1"]
    person2 = request.form["person2"]
    group_type = request.form["group_type"]
    relation_type = request.form["relation_type"]

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="user",
        database="social_community"
    )

    cursor = conn.cursor()

    # Insert person1 if not exists
    cursor.execute("SELECT person_id FROM person WHERE person_name=%s", (person1,))
    result1 = cursor.fetchone()

    if result1:
        p1_id = result1[0]
    else:
        cursor.execute("INSERT INTO person (person_name) VALUES (%s)", (person1,))
        p1_id = cursor.lastrowid

    # Insert person2 if not exists
    cursor.execute("SELECT person_id FROM person WHERE person_name=%s", (person2,))
    result2 = cursor.fetchone()

    if result2:
        p2_id = result2[0]
    else:
        cursor.execute("INSERT INTO person (person_name) VALUES (%s)", (person2,))
        p2_id = cursor.lastrowid

    # Insert relationship
    try:
        cursor.execute("""
            INSERT INTO relationships (person1_id, person2_id, group_type, relation_type)
            VALUES (%s, %s, %s, %s)
        """, (p1_id, p2_id, group_type, relation_type))
    except:
        pass

    conn.commit()
    cursor.close()
    conn.close()

    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)