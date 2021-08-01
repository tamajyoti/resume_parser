from utils import *
from constants import EDUCATION


class Resume:

    def __init__(self, path):
        self.resume_text = get_resume_text(path)
        self.mobile_number = extract_mobile_number(self.resume_text)
        self.email = extract_email(self.resume_text)
        self.spacy_text, self.matcher = covert_text_spacy(self.resume_text, resume_flag=True)
        self.name = extract_name(self.spacy_text, self.matcher)
        self.city = extract_city(self.spacy_text)
        self.all_experience = extract_experience(self.spacy_text)
        self.all_education = extract_education(self.all_experience, EDUCATION)
        self.all_prof_orgs = extract_professional_experience(self.all_experience, EDUCATION)
        self.skill_dump = extract_skills(self.spacy_text)
        self.doc_embeddings = get_embeddings(self.skill_dump)


class Job:
    def __init__(self, path):
        self.job_text = get_job_description(path)
        self.spacy_text = covert_text_spacy(self.job_text, resume_flag=False)
        self.doc_embeddings = get_embeddings([sentence for sentence in self.spacy_text.sents])


