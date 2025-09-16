
# Senior AI/ML Developer Roadmap — 48 Weeks

<details>
<summary>Phase 0 — Setup (Week 0)</summary>

<details>
<summary>Week 0 — Python & Environment Setup</summary>

| Training / Book / Video          | Alternatives                  | Author / Platform | Task / Project                                                        | Notes / Focus                                       | Learning Outcome                                | Link                                                              | Pros                        | Cons                                | Difficulty | Hands-On Focus | Recommendation           |
| -------------------------------- | ----------------------------- | ----------------- | --------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------- | --------------------------- | ----------------------------------- | ---------- | -------------- | ------------------------ |
| Quick Python & environment setup | N/A                           | Self-study        | Install Python 3.10+, Conda/venv, VS Code, Git, Docker, Jupyter/Colab | Ensure a fully functional AI/ML dev environment     | Ready to code and run ML projects locally       | N/A                                                               | • Fast setup <br>• Flexible | • Self-paced may require discipline | Easy       | Medium         | Mandatory                |
| Linux / CLI refresher            | Udemy: "Linux for Developers" | Udemy / YouTube   | Practice basic commands, navigation, scripts                          | Fast track Linux basics needed for dev & deployment | Able to navigate, run scripts, and manage files | [Udemy Linux](https://www.udemy.com/course/linux-for-developers/) | • Practical for dev ops     | • Some commands may be repetitive   | Easy       | Medium         | Optional but recommended |

</details>
</details>

<details>
<summary>Phase 1 — Foundations (Weeks 1–6)</summary>

<details>
<summary>Week 1–2 — Python for Data Analysis</summary>

| Training / Book / Video                     | Alternatives                            | Author / Platform            | Task / Project                                                              | Notes / Focus                                                     | Learning Outcome                                   | Link                                                                                                                                                                          | Pros                                         | Cons                                              | Difficulty | Hands-On Focus | Recommendation |
| ------------------------------------------- | --------------------------------------- | ---------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------- | ---------- | -------------- | -------------- |
| Python for Data Analysis (Book) — preferred | Udemy: Python for Data Science Bootcamp | Wes McKinney / Jose Portilla | Practice advanced Pandas: `groupby`, `pivot_table`, `merge`, vectorized ops | Focus on real-world dataset manipulation, efficient data handling | Can clean, transform, and analyze complex datasets | [Book](https://www.oreilly.com/library/view/python-for-data/9781491957653/) <br> [Udemy](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/) | • Strong Pandas coverage <br>• Real datasets | • Less interactive <br>• Requires self-discipline | Medium     | High           | Mandatory      |

</details>

<details>
<summary>Week 3–4 — Mathematics for Machine Learning</summary>

| Training / Book / Video                                     | Alternatives                         | Author / Platform                      | Task / Project                                                  | Notes / Focus                                              | Learning Outcome                                           | Link                                                                              | Pros                                   | Cons          | Difficulty | Hands-On Focus | Recommendation   |
| ----------------------------------------------------------- | ------------------------------------ | -------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------- | ------------- | ---------- | -------------- | ---------------- |
| Mathematics for Machine Learning Specialization — preferred | Khan Academy: Linear Algebra & Stats | Imperial College London / Khan Academy | Implement PCA, matrix decompositions, gradient descent in NumPy | Focused on vector calculus, matrix ops, probability for ML | Able to explain & implement mathematical foundations of ML | [Coursera](https://www.coursera.org/specializations/mathematics-machine-learning) | • Strong foundation <br>• Step-by-step | • Less coding | Medium     | Medium         | Mandatory for ML |

</details>

<details>
<summary>Week 5–6 — Linear Algebra & Stats</summary>

| Training / Book / Video                                       | Alternatives                                            | Author / Platform          | Task / Project                | Notes / Focus                                             | Learning Outcome                                              | Link                                                                                                                                                             | Pros                           | Cons                                      | Difficulty | Hands-On Focus | Recommendation |
| ------------------------------------------------------------- | ------------------------------------------------------- | -------------------------- | ----------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ----------------------------------------- | ---------- | -------------- | -------------- |
| Essence of Linear Algebra (YouTube) + Khan Academy Statistics | LinkedIn Learning: Data Science Foundations: Statistics | 3Blue1Brown / Khan Academy | Watch videos, solve exercises | Visual intuition for linear algebra & stats; fast refresh | Able to understand vector spaces, eigenvectors, distributions | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) <br> [Khan Academy](https://www.khanacademy.org/math/statistics-probability) | • Visual, intuitive <br>• Free | • Less structured <br>• Not comprehensive | Medium     | Medium         | Recommended    |

</details>

<details>
<summary>Week 6 — Checkpoint Project</summary>

| Training / Book / Video                                        | Alternatives            | Author / Platform | Task / Project                                          | Notes / Focus                                     | Learning Outcome                             | Link                                                                  | Pros                                    | Cons             | Difficulty | Hands-On Focus | Recommendation |
| -------------------------------------------------------------- | ----------------------- | ----------------- | ------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------- | ---------------- | ---------- | -------------- | -------------- |
| Build a data pipeline from a SQL database and create dashboard | Streamlit / Gradio demo | Self-study        | Transform data, visualize, create interactive dashboard | Integrates SQL knowledge + Pandas + visualization | Demonstrates end-to-end data handling skills | [Streamlit](https://streamlit.io/) <br> [Gradio](https://gradio.app/) | • End-to-end integration <br>• Hands-on | • Time-consuming | Medium     | High           | Mandatory      |

</details>
</details>

<details>
<summary>Phase 2 — Core Machine Learning & Advanced ML (Weeks 7–20)</summary>

<details>
<summary>Week 7–8 — Hands-On Machine Learning</summary>

| Training / Book / Video                                                     | Alternatives                | Author / Platform      | Task / Project                                      | Notes / Focus                     | Learning Outcome                                 | Link                                                                                                                                                     | Pros                                              | Cons                                             | Difficulty  | Hands-On Focus | Recommendation                 |
| --------------------------------------------------------------------------- | --------------------------- | ---------------------- | --------------------------------------------------- | --------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------ | ----------- | -------------- | ------------------------------ |
| Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — preferred | Udemy: Machine Learning A-Z | Aurélien Géron / Udemy | Build regression, classification, clustering models | Practical, code-first ML pipeline | Implement real ML workflows and model evaluation | [Book](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291/) <br> [Udemy](https://www.udemy.com/course/machinelearning/) | • Very practical <br>• Covers modern ML libraries | • Steep learning curve                           | Medium-High | High           | Best for hands-on coders       |
| Andrew Ng Machine Learning Specialization (Coursera) — alternative          | N/A                         | Andrew Ng / Coursera   | Implement ML algorithms with Octave / Python        | Step-by-step theory + quizzes     | Strong ML intuition before coding                | [Coursera](https://www.coursera.org/specializations/machine-learning)                                                                                    | • Guided, gentle introduction <br>• Strong theory | • Less modern ML coverage <br>• Limited hands-on | Medium      | Medium         | Best for theory-first learners |

</details>

<details>
<summary>Week 9–12 — Practical Deep Learning</summary>

| Training / Book / Video                                | Alternatives                          | Author / Platform         | Task / Project                        | Notes / Focus                                         | Learning Outcome                            | Link                                                                                                       | Pros                                                       | Cons                                      | Difficulty  | Hands-On Focus | Recommendation                 |
| ------------------------------------------------------ | ------------------------------------- | ------------------------- | ------------------------------------- | ----------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------- | ----------- | -------------- | ------------------------------ |
| fast.ai Practical Deep Learning for Coders — preferred | Coursera Deep Learning Specialization | Jeremy Howard / Andrew Ng | Train CNNs, RNNs, Transformers        | Top-down, code-first deep learning; focus on projects | Able to build and train deep models quickly | [fast.ai](https://course.fast.ai/) <br> [Coursera](https://www.coursera.org/specializations/deep-learning) | • Very hands-on <br>• Rapid prototyping <br>• Uses PyTorch | • Requires Python & Math knowledge        | High        | Very High      | Best for coding-first learners |
| Coursera Deep Learning Specialization — alternative    | N/A                                   | Andrew Ng / Coursera      | Follow theory + implement DL in Keras | Step-by-step guidance                                 | Understand DL fundamentals                  | [Coursera](https://www.coursera.org/specializations/deep-learning)                                         | • Strong theory coverage <br>• Step-by-step guidance       | • Less hands-on <br>• Slower for projects | Medium-High | Medium         | Best for theory-first learners |

</details>

<details>
<summary>Week 13–16 — Neural Networks from Scratch</summary>

| Training / Book / Video                             | Alternatives                  | Author / Platform       | Task / Project                        | Notes / Focus                                           | Learning Outcome                                             | Link                                                                                                                                          | Pros                                                         | Cons                                                | Difficulty  | Hands-On Focus | Recommendation         |
| --------------------------------------------------- | ----------------------------- | ----------------------- | ------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------- | ----------- | -------------- | ---------------------- |
| Neural Networks: Zero to Hero (YouTube) — preferred | Udemy: Deep Learning Bootcamp | Andrej Karpathy / Udemy | Implement NN from scratch using NumPy | Build intuition on backpropagation, layers, activations | Understand neural network fundamentals from first principles | [YouTube](https://www.youtube.com/playlist?list=PLjJh1vlSEYgv3u2khHca7g0q0_zyFtx9c) <br> [Udemy](https://www.udemy.com/course/deep-learning/) | • Learn from scratch <br>• Strong intuition <br>• Free video | • Less structured <br>• Slower for full DL coverage | Medium-High | High           | Great for fundamentals |

</details>

<details>
<summary>Week 17–20 — Deep Learning Specialization & Checkpoint</summary>

| Training / Book / Video                                | Alternatives                                   | Author / Platform    | Task / Project                       | Notes / Focus                                         | Learning Outcome                                        | Link                                                               | Pros                                                 | Cons                                             | Difficulty | Hands-On Focus | Recommendation       |
| ------------------------------------------------------ | ---------------------------------------------- | -------------------- | ------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------ | ---------- | -------------- | -------------------- |
| Deep Learning Specialization (Coursera) — supplemental | N/A                                            | Andrew Ng / Coursera | Cover theory, regularization, tuning | Complements fast.ai & Karpathy by filling theory gaps | Solid theoretical grounding + practical implementations | [Coursera](https://www.coursera.org/specializations/deep-learning) | • Strong theory <br>• Step-by-step learning          | • Less hands-on coding <br>• Slower project pace | Medium     | Medium         | Combine with fast.ai |
| Week 20 Checkpoint                                     | Re-implement k-means or simple NN from scratch | N/A                  | Self-study                           | Only NumPy; no ML library                             | Demonstrates deep understanding beyond library usage    | Ability to implement ML/DL algorithms manually                     | • Practice critical for understanding <br>• Flexible | • Requires discipline                            | High       | Very High      | Mandatory            |

</details>
</details>

<details>
<summary>Phase 3 — LLM Engineering & NLP (Weeks 21–32)</summary>

| Training / Book / Video                                    | Alternatives           | Author / Platform                          | Task / Project                       | Notes / Focus                                         | Learning Outcome                             | Link                                                                                                                                                          | Pros                                                | Cons                             | Difficulty | Hands-On Focus | Recommendation                     |
| ---------------------------------------------------------- | ---------------------- | ------------------------------------------ | ------------------------------------ | ----------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------- | ---------- | -------------- | ---------------------------------- |
| Hugging Face Course + CS224N — preferred                   | Udemy NLP course       | Hugging Face / Stanford                    | Fine-tune transformers, NLP projects | Transformers, embeddings, tokenization, attention     | Able to implement transformer models for NLP | [Hugging Face](https://huggingface.co/learn) <br> [CS224N](http://web.stanford.edu/class/cs224n/)                                                             | • Very practical <br>• Hands-on projects <br>• Free | • Requires Python & ML knowledge | High       | Very High      | Best for coding-first LLM learners |
| Full Stack LLM Bootcamp — preferred                        | LangChain short course | Full Stack Deep Learning / DeepLearning.AI | Build end-to-end LLM apps            | Prompt engineering, embedding pipelines, vector DB    | Able to build full-stack LLM applications    | [FS Deep Learning](https://fullstackdeeplearning.com/) <br> [LangChain](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) | • End-to-end LLM <br>• Hands-on                     | • Paid <br>• Requires coding     | High       | Very High      | Best for hands-on LLM projects     |
| Building Advanced RAG Applications (Coursera) + LlamaIndex | OpenAI Cookbook        | DeepLearning.AI / LlamaIndex               | Build multi-agent RAG assistant      | Retrieval-augmented generation, embeddings, vector DB | Able to design knowledge-based AI systems    | [Coursera](https://www.coursera.org/) <br> [LlamaIndex Docs](https://gpt-index.readthedocs.io/en/latest/)                                                     | • Practical RAG implementation                      | • Some prep work required        | High       | Very High      | Best for senior LLM engineering    |

</details>

<details>
<summary>Phase 4 — MLOps & Production Engineering (Weeks 33–48)</summary>

| Training / Book / Video                           | Alternatives                               | Author / Platform           | Task / Project                                                          | Notes / Focus                                                | Learning Outcome                                         | Link                                                                        | Pros                                         | Cons                                       | Difficulty | Hands-On Focus | Recommendation                    |
| ------------------------------------------------- | ------------------------------------------ | --------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------ | ---------- | -------------- | --------------------------------- |
| Designing Data-Intensive Applications — preferred | LinkedIn Learning: Data Architecture       | Martin Kleppmann / O’Reilly | Read book + exercises                                                   | Distributed systems, pipelines, consistency, fault tolerance | Design scalable, production-ready ML systems             | [Book](https://dataintensive.net/)                                          | • Strong design principles                   | • Theory-heavy                             | Medium     | Medium         | Recommended                       |
| MLOps                                             | Machine Learning Operations Specialization | Duke University / Coursera  | Build pipeline with Docker, Kubernetes, MLflow, DVC                     | CI/CD, monitoring, versioning, cloud deployment              | Deploy production ML pipelines efficiently               | [Coursera](https://www.coursera.org/specializations/mlops-machine-learning) | • Hands-on <br>• Covers production pipelines | • Paid <br>• Some tools setup needed       | High       | High           | Mandatory for production-ready ML |
| Capstone & Portfolio                              | N/A                                        | Self-practice               | Build public end-to-end project (e.g., RAG system or anomaly detection) | Apply everything learned, document design, create CI/CD      | Demonstrates senior-level, business-ready AI/ML skillset | N/A                                                                         | • Showcases skills <br>• Portfolio ready     | • Time-consuming <br>• Requires discipline | High       | Very High      | Mandatory                         |

</details>

---
