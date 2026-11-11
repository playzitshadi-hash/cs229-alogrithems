import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn

from sentence_transformers import SentenceTransformer

documents = [
    # --- Education & Admissions ---
    "Harvard provides financial aid to eligible students.",
    "Students can prepare for the SAT using official practice tests.",
    "Scholarships can help students pay university tuition.",
    "The Common App allows students to apply to multiple colleges at once.",
    "Letters of recommendation give insight into a student's character.",
    "Extracurricular activities can strengthen a college application.",
    "Community colleges offer affordable pathways to higher education.",
    "Standardized tests measure skills in math and reading.",
    "Financial aid packages may include grants, loans, and work-study.",
    "Early decision applications require a binding commitment to enroll.",

    # --- Technology & Computer Science ---
    "Python is commonly used for artificial intelligence.",
    "Machine learning models learn patterns from data.",
    "Neural networks are inspired by the structure of the brain.",
    "Cloud computing provides on-demand access to computing resources.",
    "Version control systems track changes to source code.",
    "Encryption protects data by converting it into unreadable form.",
    "APIs allow different software systems to communicate with each other.",
    "Open-source software can be freely used, modified, and shared.",
    "Databases store and organize large amounts of structured information.",
    "Cybersecurity focuses on protecting systems from digital attacks.",

    # --- AI & Data ---
    "Supervised learning requires labeled training data.",
    "Large language models are trained on vast amounts of text.",
    "Overfitting occurs when a model memorizes training data too closely.",
    "Data cleaning removes errors and inconsistencies before analysis.",
    "Reinforcement learning trains agents using rewards and penalties.",
    "Natural language processing helps computers understand human text.",
    "Model evaluation measures how well a system performs on new data.",

    # --- General Knowledge ---
    "The human brain contains roughly eighty-six billion neurons.",
    "Regular exercise supports both physical and mental health.",
    "The scientific method relies on observation and experimentation.",
]

encoder = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = encoder.encode(
    documents,
    normalize_embeddings=True
)
query = input("Write ")
query_embedding = encoder.encode(
    query,
    normalize_embeddings=True
)
sim = query_embedding @ document_embeddings.T
sorted_ar = np.argsort(sim)[::-1]
sort = sorted_ar[:3]
for index in sort:
    print(documents[index])
    print(sim[index])
