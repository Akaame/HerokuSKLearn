
from flask import Flask, request, jsonify, render_template
# Flask: uygulamayi olusturmak icin
# request: gelen veriyi almak icin
# jsonify: JSON formatinda cikti dondurmek icin
# render_template: HTML sayfasi "render"lamak icin
from sklearn.externals import joblib # Pickle ile deserialization

app = Flask(__name__) # uygulama olustur
clr = joblib.load("svc_model.pkl") # modeli oku

@app.route('/predict', methods=["POST"])
def predict():
    # modeli predict icinde okursaniz 
    # her istek geldiginde dosyadan yeni bir model olusturulacak
    # bunu engellemek icin global tanimladik
    global clr
    data = request.form # form verisini al
    data = [data["data_1"], data["data_2"], data["data_3"], data["data_4"] ] # form verisi
    # Burada gelen veriyi "sanitize" etmeniz sizin faydaniza olur.
    # Kullanici girdisine asla guvenmeyin.
    res = list(clr.predict([data])) 
    # veriyi [ [..degerler] ] seklinde modele verip tahmin aldik
    return jsonify([res]) # JSON formatinda dondurelim

@app.route('/', methods=["GET"])
def home():
    # arayuzu koddan ayiralim
    return render_template("home_template.html")
if __name__ == "__main__":
    app.run()
