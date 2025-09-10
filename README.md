```markdown
# Senior AI/ML Developer Roadmap (48 Weeks)

Designed for experienced developers (~18 years) to become **Senior AI/ML developers**.  
Video-first, hands-on, practical, with multiple course alternatives.

---

## Phase 0 — Setup (Week 0)
<details>
<summary>Click to expand</summary>

| Week | Course / Resource | Author / Platform | Focus / Notes | Task / Project | Outcome | Link | Alternatives |
|------|-----------------|-----------------|---------------|----------------|---------|------|-------------|
| 0 | Environment Setup | Self-guided | Python 3.10+, Conda/venv, VS Code, Git, Docker, Jupyter / Colab, basic Linux/CLI. Prepare environment to avoid issues later. | Install and configure all tools; run “Hello World” notebooks | Ready dev environment for ML & DL | [Guide](https://realpython.com/python-virtual-environments-a-primer/) | [LinkedIn Learning: Python Dev Env](https://www.linkedin.com/learning/python-essential-training-2) |

</details>

---

## Phase 1 — Foundations (Weeks 1–6)
<details>
<summary>Click to expand</summary>

| Week | Course / Resource | Author / Platform | Focus / Notes | Task / Project | Outcome | Link | Alternatives |
|------|-----------------|-----------------|---------------|----------------|---------|------|-------------|
| 1 | Python for Data Analysis | Wes McKinney / O'Reilly | Advanced Pandas & NumPy, vectorized ops, groupby, pivot tables. Focus on speed and efficiency. | Practice complex data wrangling on CSV/SQL exports | Efficient data manipulation & analysis | [Video](https://www.oreilly.com/videos/python-for-data/9781803243979/) | Udemy: [Python for Data Analysis](https://www.udemy.com/course/python-for-data-analysis/), LinkedIn Learning: [Python Data Analysis](https://www.linkedin.com/learning/python-for-data-science-essential-training-2) |
| 2 | Effective Pandas | Matt Harrison / YouTube | Optimized DataFrame operations, joins, performance tricks. Focus on large, dirty datasets. | Apply cleaning & aggregation on messy datasets | Master Pandas for ML workflows | [Video](https://www.youtube.com/watch?v=zgbUk90aQ6A) | Udemy: [Pandas Masterclass](https://www.udemy.com/course/pandas-data-analysis/), LinkedIn Learning: [Data Analysis with Pandas](https://www.linkedin.com/learning/pandas-data-analysis/) |
| 3 | Essence of Linear Algebra | 3Blue1Brown / YouTube | Visual linear algebra concepts: vectors, matrices, transformations. Build intuition for ML. | Implement small exercises in NumPy | Intuitive grasp of linear algebra | [Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Udemy: [Linear Algebra for ML](https://www.udemy.com/course/linear-algebra-for-machine-learning/), LinkedIn Learning: [Linear Algebra Foundations](https://www.linkedin.com/learning/linear-algebra-foundations) |
| 4 | Mathematics for Machine Learning | Imperial College London / Coursera | Vector calculus, matrix decompositions, linear algebra applied in ML. Focus on practical implementation. | Implement PCA/SVD from scratch in NumPy | Strong math foundation for ML | [Course](https://www.coursera.org/specializations/mathematics-machine-learning) | LinkedIn Learning: [Linear Algebra for ML](https://www.linkedin.com/learning/linear-algebra-for-machine-learning) |
| 5 | Probability & Statistics | Khan Academy | Distributions, hypothesis testing, sampling, applied to ML metrics. Focus on understanding evaluation metrics. | Apply probability metrics on real datasets | Able to compute precision, recall, AUC | [Khan Academy](https://www.khanacademy.org/math/statistics-probability) | LinkedIn Learning: [Statistics for Data Science](https://www.linkedin.com/learning/statistics-foundations-1), Udemy: [Statistics for Data Science](https://www.udemy.com/course/statistics-for-data-science/) |
| 6 | Data Pipeline Mini-Project | Self-guided | Combine SQL + Python + Pandas + visualization. Focus on end-to-end data engineering basics. | Build data pipeline: SQL → Python transformations → Dashboard | End-to-end ETL & visualization experience | N/A | Udemy: [Python ETL & Data Pipelines](https://www.udemy.com/course/python-data-pipelines/) |

</details>

---

## Phase 2 — Core ML & Advanced ML (Weeks 7–20)
<details>
<summary>Click to expand</summary>

| Week | Course / Resource | Author / Platform | Focus / Notes | Task / Project | Outcome | Link | Alternatives |
|------|-----------------|-----------------|---------------|----------------|---------|------|-------------|
| 7 | Hands-On Machine Learning with Scikit-Learn | Aurélien Géron / Book + Code | Implement classical ML models: regression, trees, SVMs. Pipelines and preprocessing. Focus on code-first approach. | Build ML pipeline: data → model → evaluation → API | Practical ML implementation skills | [Book](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291) | Udemy: [Hands-On ML](https://www.udemy.com/course/hands-on-machine-learning/), LinkedIn Learning: [Machine Learning Foundations](https://www.linkedin.com/learning/machine-learning-foundations/) |
| 8 | Practical Deep Learning for Coders | fast.ai / Video | Applied DL: CNNs, RNNs, transfer learning, PyTorch. Focus on building real models quickly. | Train image classifier & text model | Real-world DL experience | [Course](https://course.fast.ai/) | Udemy: [Deep Learning with PyTorch](https://www.udemy.com/course/deep-learning-with-pytorch/), LinkedIn Learning: [Deep Learning Fundamentals](https://www.linkedin.com/learning/deep-learning-foundations) |
| 9 | Neural Networks: Zero to Hero | Andrej Karpathy / YouTube | Build NNs from scratch, backpropagation, optimization. Focus on deep conceptual understanding. | Implement feedforward NN from scratch | Deep understanding of NN internals | [Video](https://www.youtube.com/watch?v=aircAruvnKk) | N/A |
| 10 | Deep Learning Specialization | Andrew Ng / Coursera | Supplement theory: CNN, RNN, hyperparameter tuning. Focus on filling gaps from coding-first courses. | Experiment with Keras/TensorFlow | Solid theoretical foundation | [Course](https://www.coursera.org/specializations/deep-learning) | Udemy: [Deep Learning A-Z](https://www.udemy.com/course/deeplearning/), LinkedIn Learning: [Deep Learning Foundations](https://www.linkedin.com/learning/deep-learning-foundations) |
| 11–12 | ML Deployment Mini-Project | Self-guided | Focus on deploying ML/DL models using Gradio / Streamlit / Flask. End-to-end experience. | Deploy trained ML/DL model as API | End-to-end project deployment skills | N/A | Udemy: [Python Flask API Deployment](https://www.udemy.com/course/python-flask-api/) |
| 13–14 | Model Optimization & Hyperparameters | Self-guided | Grid search, CV, regularization, early stopping. Focus on practical performance tuning. | Tune existing models & compare performance | ML model evaluation & optimization | N/A | Udemy: [Hyperparameter Tuning in Python](https://www.udemy.com/course/hyperparameter-tuning-in-python/) |
| 15–16 | Capstone ML/DL Project | Self-guided | Integrate ML + DL pipelines. Focus on portfolio-ready, real-world problem. | Build production-ready ML/DL project | Portfolio-ready ML/DL system | N/A | Udemy: [End-to-End ML Project](https://www.udemy.com/course/end-to-end-machine-learning-project/) |
| 17–20 | Optional Advanced Theories | DeepLearning.AI / Coursera | Fill gaps from applied DL courses. Focus on advanced architectures (ResNet, Transformers). | Implement complex architectures | Deep learning mastery for professional use | [Course](https://www.deeplearning.ai/) | LinkedIn Learning: [Advanced Deep Learning](https://www.linkedin.com/learning/advanced-deep-learning) |

</details>

---

## Phase 3 — NLP, LLMs & Agents (Weeks 21–36)
<details>
<summary>Click to expand</summary>

| Week | Course / Resource | Author / Platform | Focus / Notes | Task / Project | Outcome | Link | Alternatives |
|------|-----------------|-----------------|---------------|----------------|---------|------|-------------|
| 21–24 | Hugging Face Course | Hugging Face / Video & Labs | Transformers, tokenizers, fine-tuning LLMs. Focus on applied NLP. | Fine-tune transformer on domain data | Hands-on NLP & LLM skills | [Course](https://huggingface.co/learn) | Udemy: [Hugging Face Transformers](https://www.udemy.com/course/hugging-face-transformers/), LinkedIn Learning: [Applied NLP](https://www.linkedin.com/learning/natural-language-processing) |
| 21–24 | CS224N: NLP with Deep Learning | Stanford / Video & Assignments | Transformers theory, embeddings, attention. Deep conceptual understanding. | Implement small transformer-based project | Solid theoretical & coding skills in NLP | [Course](http://web.stanford.edu/class/cs224n/) | N/A |
| 25–28 | Full Stack LLM Bootcamp | Full Stack Deep Learning
```
