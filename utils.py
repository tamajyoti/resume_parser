import PyPDF2
import re

import numpy as np
import spacy
from spacy.matcher import Matcher
from sentence_transformers import SentenceTransformer


def get_resume_text(resume_path):
    pdf_file_obj = open(resume_path, 'rb')

    # creating a pdf reader object
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    all_text = []
    for i in range(pdf_reader.numPages):
        page_obj = pdf_reader.getPage(i)
        text_data = page_obj.extractText()
        all_text.append(text_data)

    return "".join(all_text).replace("\n", "")


def get_job_description(jd_path):
    f = open("resumes/Senior Risk Modelling Analyst", "r")
    job_text = f.read().replace("\n", "")
    return job_text


def extract_mobile_number(text):
    """
    Helper function to extract mobile number from text
    :param text: plain text extracted from resume file
    :return: string of extracted mobile numbers
    """
    # Found this complicated regex on : https://zapier.com/blog/extract-links-email-phone-regex/
    phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1['
                                  r'02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9]['
                                  r'02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9]['
                                  r'02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'),
                       text)
    if phone:
        number = ''.join(phone[0])
        if len(number) > 10:
            return '+' + number
        else:
            return number


def extract_email(text):
    """
    Helper function to extract email id from text
    :param text: plain text extracted from resume file
    """
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None


def covert_text_spacy(text, resume_flag):
    nlp = spacy.load("en_core_web_sm")
    spacy_text = nlp(text)
    matcher = Matcher(nlp.vocab, validate=True)
    if resume_flag:
        return spacy_text, matcher
    else:
        return spacy_text


def extract_name(spacy_text, matcher):
    # identify a pattern
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    ]
    matcher.add("process_1", patterns)
    matches = matcher(spacy_text)
    all_match = []
    for _, start, end in matches:
        all_match.append(spacy_text[start:end].text)

    if all_match:
        return all_match[0]
    else:
        return ""


def extract_experience(spacy_text):
    all_experience = []
    for sent in spacy_text.sents:
        results = re.findall('(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\s-]\d{2,4}', sent.text.lower())
        if results:
            org_list = []
            for token in sent.ents:
                if token.label_ == "ORG":
                    org_list.append(token)
            all_experience.append([results, org_list])
    return all_experience


def extract_education(experience_list, education_list):
    all_education = []
    for val in experience_list:
        education_tokens = [tokens for education in education_list for tokens in val[1] if
                            education in tokens.text.lower()]
        if education_tokens:
            education_details = {"time_of_education": val[0],
                                 "institutes": education_tokens}
            all_education.append(education_details)
    return all_education


def extract_professional_experience(experience_list, education_list):
    prof_experience = []
    for val in experience_list:
        experience_tokens = [tokens for education in education_list for tokens in val[1] if
                             education not in tokens.text.lower()]
        if experience_tokens:
            education_details = {"time_of_experience": val[0],
                                 "organizations": experience_tokens}
            prof_experience.append(education_details)
    return prof_experience


def extract_skills(spacy_text):
    skill_list = []
    for i, chunk in enumerate(spacy_text.noun_chunks):
        if len(chunk) > 2 and "experience" in chunk.text.lower():
            exp_index = i
            break
    for i, chunk in enumerate(spacy_text.noun_chunks):
        if i >= exp_index and len(chunk) >= 2:
            skill_list.append(chunk)

    return skill_list


def get_embeddings(sentence_list):
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    embeddings = model.encode(sentence_list)
    text_encoding = np.mean(embeddings, axis=0)
    return text_encoding
