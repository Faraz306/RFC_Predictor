import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="YF Predictor", layout="centered")


@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("train.csv")
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    return df


df = load_and_clean_data()

# 2. Train the Machine Learning Model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']
mlmodel = RandomForestClassifier(random_state=42)
mlmodel.fit(X, y)

# 3. Main App Layout
st.title("Yamaan Faraz YF Predictor (Titanic Dataset)")

Name = st.text_input("Name", placeholder="Enter your Name")
age = st.slider("Age", 0, 100, 25)
pclass = st.slider("Class", 1, 3, 3)
gender = st.selectbox("Gender", ["Male", "Female"])
sibling = st.slider("Sibling", 0, 5, 0)
children = st.slider("Children", 0, 5, 0)
Fare = st.slider("Fare", 0, 500, 32)

animation_container = st.empty()

# 4. Process Prediction & Handle Outcomes
if st.button("Predict"):
    if Name.strip():
        gender_encoded = 1 if gender == "Male" else 0
        user_data = pd.DataFrame([[pclass, gender_encoded, age, sibling, children, Fare]], columns=features)

        prediction = mlmodel.predict(user_data)[0]

        with animation_container:
            # --- STANDARD SURVIVAL OUTCOME ---
            if prediction == 1:
                st.balloons()
                happy_confetti = """
                <div style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:99999;"><canvas id="c"></canvas></div>
                <script>
                var c=document.getElementById("c"),ctx=c.getContext("2d"),p=[];
                c.width=window.innerWidth;c.height=window.innerHeight;
                for(let i=0;i<100;i++)p.push({x:c.width/2,y:c.height,vx:Math.random()*20-10,vy:Math.random()*-20-10,color:"hsl("+360*Math.random()+",100%,50%)",r:Math.random()*6+4});
                function draw(){
                    ctx.clearRect(0,0,c.width,c.height);
                    p.forEach(q=>{q.x+=q.vx;q.y+=q.vy;q.vy+=0.5;ctx.fillStyle=q.color;ctx.beginPath();ctx.arc(q.x,q.y,q.r,0,2*Math.PI);ctx.fill();});
                    requestAnimationFrame(draw);
                }draw();
                </script>
                """
                st.components.v1.html(happy_confetti, height=0)
                st.success(f"🎉 Congratulations {Name}, you would have survived the Titanic!")

            # --- STANDARD FATALITY OUTCOME ---
            else:
                sad_rain = """
                <div style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:99999;"><canvas id="r"></canvas></div>
                <script>
                var r=document.getElementById("r"),ctx=r.getContext("2d"),drops=[];
                r.width=window.innerWidth;r.height=window.innerHeight;
                for(let i=0;i<50;i++)drops.push({x:Math.random()*r.width,y:Math.random()*r.height,v:Math.random()*5+5});
                function draw(){
                    ctx.clearRect(0,0,r.width,r.height);ctx.font="40px Arial";
                    drops.forEach(d=>{ctx.fillText("😭",d.x,d.y);d.y+=d.v;if(d.y>r.height){d.y=-40;d.x=Math.random()*r.width;}});
                    requestAnimationFrame(draw);
                }draw();
                </script>
                """
                st.components.v1.html(sad_rain, height=0)
                st.error(f"😔 Sorry {Name}, you would not have survived the Titanic.")
    else:
        st.error("Please enter your name first!")
