import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cs
import glob
from data_objects import Resume, Job


class Model:

    def __init__(self, resume_path, jd_path, top_n):
        self.top_n = top_n
        resumes = []
        for file in glob.glob(resume_path + "*.pdf"):
            resume_object = Resume(file)
            resumes.append(resume_object)

        self.resumes = resumes
        self.job = Job(jd_path)
        self.resume_job_sim = self.get_cosine_similarity()
        self.output_dataframe = self.get_dataframe()

    def get_cosine_similarity(self):
        resume_job_sim = []
        for resume in self.resumes:
            cos_sim = cs(resume.doc_embeddings.reshape(1, -1), self.job.doc_embeddings.reshape(1, -1))
            resume_job_sim.append(cos_sim[0][0])

        return resume_job_sim

    def get_dataframe(self):
        df_resume = pd.DataFrame({"resume_objects": self.resumes, "similarity": self.resume_job_sim})
        df_resume = df_resume.sort_values("similarity", ascending=False)
        df_resume["rank"] = df_resume["similarity"].rank()
        df_resume = df_resume[:self.top_n]
        df_resume["name"] = df_resume.apply(lambda row: row.resume_objects.name, axis=1)
        df_resume["mobile"] = df_resume.apply(lambda row: row.resume_objects.mobile_number, axis=1)
        df_resume["email"] = df_resume.apply(lambda row: row.resume_objects.email, axis=1)
        df_resume["name"] = df_resume.apply(lambda row: row.resume_objects.name, axis=1)
        df_resume["mobile"] = df_resume.apply(lambda row: row.resume_objects.mobile_number, axis=1)
        df_resume["email"] = df_resume.apply(lambda row: row.resume_objects.email, axis=1)
        df_resume["city"] = df_resume.apply(lambda row: row.resume_objects.city.text, axis=1)
        # education experience and skills cleaned for final reporting
        df_resume["education"] = df_resume.apply(
            lambda row: set([exp for val in row.resume_objects.all_education for vals in val.values() for exp in vals]),
            axis=1)
        df_resume["experience"] = df_resume.apply(
            lambda row: set([exp for val in row.resume_objects.all_prof_orgs for vals in val.values() for exp in vals]),
            axis=1)
        df_resume["skills"] = df_resume.apply(lambda row: [skill.text for skill in row.resume_objects.skill_dump],
                                              axis=1)
        df_resume = df_resume.drop(["resume_objects"], axis=1)
        df_resume = df_resume.reset_index(drop=True)

        return df_resume
