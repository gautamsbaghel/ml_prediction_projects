from flask import Flask,render_template,request
app = Flask(__name__)
import pickle



#open file where you stored pickled data
file=open('salary.pkl', 'rb')
lr = pickle.load(file)  #used to read data from file
file.close()



@app.route('/', methods=["GET","POST"])#if you write only method instead of methods it will give error
def salary():
    if request.method == "POST":
        myDict = request.form
        experience = float(myDict['experience'])

        inputFeatures = [experience]
        Estimated_Salary = lr.predict([inputFeatures])
        print(Estimated_Salary)
        return render_template('show.html', inf=int(Estimated_Salary)) 
    return render_template('index.html')  #it will open our index file in web local host


if __name__=="__main__":  #this will generate localhost
    app.run(debug=True)