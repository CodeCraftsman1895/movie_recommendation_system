# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Natural Language Processing (NLP) techniques.  
This project suggests top 5 movies similar to a given title by analyzing metadata such as genres, keywords, cast, and overview.

---

## 🚀 Live Demo
*(Add your deployed link here once hosted)*  
🔗 https://your-app-link.com  

---

## 📸 Screenshots
*(Add screenshots of your UI or output here)*  

---

## ✨ Features
- 🎯 Recommend similar movies based on content  
- 🧠 NLP-based text preprocessing (tokenization, stopword removal, stemming)  
- 🔍 Feature extraction using TfidfVectorizer  
- 📊 Cosine similarity for finding closest matches  
- ⚡ Fast and efficient recommendations  
- 📁 Clean and modular project structure  

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK  
- **Machine Learning:** TfidfVectorizer, Cosine Similarity  
- **Framework:**  Streamlit 

---

## 📂 Project Structure

movie_recommendation_system/
│
├── app.py
├── data.pkl
├── similarity.pkl
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── MRS(content based).ipynb
│
├── data/
│   └── [kaggle Dataset Link](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
│
└── assets/
    └── Screenshot.png

---

## 📓 Notebooks

- **MRS(content based).ipynb**  
  - Data cleaning, handling missing values, punctuation removal, tokenization , Creating tags, combining features, vectorization using TfidfVectorizer, Cosine similarity computation and recommendation logic.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
git clone https://github.com/your-username/movie-recommender.git  
cd movie-recommender  

### 2️⃣ Create Virtual Environment (Recommended)
python -m venv venv  
venv\Scripts\activate     # Windows  
source venv/bin/activate  # Mac/Linux  

### 3️⃣ Install Dependencies
pip install -r requirements.txt  

### 4️⃣ Run the Application
python app.py  

---

## 🌐 Deployment

### 🔹 Option 1: Render
1. Create a new Web Service on Render  
2. Connect your GitHub repository  
3. Build command:  
   pip install -r requirements.txt  
4. Start command:  
   python app.py  

---

### 🔹 Option 2: Railway
- Connect GitHub repo  
- Add environment variables (if needed)  
- Deploy directly  

---

### 🔹 Option 3: Vercel (for API-based backend)
- Use serverless functions  
- Connect repo and deploy  

---

## 📊 Dataset

- **Dataset Name:** The Movies Dataset  
- **Source:** https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset  

⚠️ Note:  
The dataset is not included in this repository due to size limitations.  
Please download it manually and place it inside the `data/` folder.

---

## 📈 How It Works

1. **Data Preprocessing**
   - Handle missing values  
   - Remove punctuation  
   - Tokenization and cleaning  

2. **Feature Engineering**
   - Combine genres, keywords, cast, and overview into tags  
   - Convert text into numerical vectors using TfidfVectorizer  

3. **Similarity Calculation**
   - Compute cosine similarity between movie vectors  

4. **Recommendation**
   - Fetch top similar movies based on similarity scores  

---

## 🧪 Example

**Input:**  
Iron Man  

**Output:**  
- Iron Man 3  
- Iron Man 2  
- Thor: Ragnarok  
- Ant-Man  
- Doctor Strange  

---

## 🔮 Future Improvements
- 🤖 Add collaborative filtering  
- 🌐 Build a full-stack web interface  
- 🔍 Improve search with fuzzy matching  
- ⚡ Use vector databases (FAISS)  
- 🧠 Add embeddings (BERT / OpenAI)  
- 👤 User authentication & personalization  

---

## 🤝 Contributing
Contributions are welcome!

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Submit a pull request  

---

## 📜 License
This project is licensed under the MIT License.

---

## 👤 Author
**Bandi Dinesh Chowdary**  

- GitHub: https://github.com/CodeCraftsman1895  
- LinkedIn: *https://www.linkedin.com/in/dinesh-chowdary-profile/*  

---

## ⭐ Acknowledgements
- Kaggle for the dataset  
- Scikit-learn for machine learning tools  
- Open-source community  

---

## 💡 Note
This project is built for learning and demonstration purposes and can be extended into a production-level recommendation system.