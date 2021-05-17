from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

#open file where you stored pickled data
file=open('model.pkl', 'rb')
clf=pickle.load(file)  #used to read data from file
file.close()

@app.route('/', methods=["GET", "POST"])#if you write only method instead of methods it will give error
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])

    #code for inference
        inputFeatures = [fever,pain,age,runnyNose,diffBreath]
        infection_prob = clf.predict_proba([inputFeatures])[0][1]
        print(infection_prob)
        return render_template('show.html', inf=round(infection_prob*100)) 
    return render_template('index.html')    #it will open our index file in web local host
    #return 'Hello, World!' + str(infection_prob)

if __name__=="__main__": #this line generate localhost
    app.run(debug=True)