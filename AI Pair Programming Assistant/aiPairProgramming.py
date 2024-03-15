# -*- coding: utf-8 -*-
"""
CIS 553 (FALL 2023) PROJECT

GROUP: CODEFORMERS
-----------------------
| AI PAIR PROGRAMMING |
-----------------------

@author: Nithesh Veerappa

On the main web app display, the user is presented with two options:
    - Enter an existing API key if he has one:
        aiPairProgObj = AI_PAIR_PROGRAMMING()

    - Generate a new API key:
        aiPairProgObj = AI_PAIR_PROGRAMMING(apiExists = False)

"""
import os
import webbrowser
import google.generativeai as LLM
from google.api_core import client_options as client_options_lib
from google.api_core import retry
# from flask import Flask, render_template, request

# app = Flask(__name__)

class AI_PAIR_PROGRAMMING():
    def __init__(self):
        self.api_key = None
        self.LLM_model = None
        self.prompt = ""
        self.generated_text = ""

    def configureAPI(self, apiExists):
        if not apiExists:
            webbrowser.open("https://developers.generativeai.google/products/palm")
        else:
            # try:
            LLM.configure(api_key=self.api_key,
                        transport="rest",
                        client_options=client_options_lib.ClientOptions(api_endpoint=os.getenv("GOOGLE_API_BASE")))
            # except google_exceptions.GoogleAPICallError as e:
                # return False
    
            models = [m for m in LLM.list_models() if 'generateText' in m.supported_generation_methods]
            self.LLM_model = models[0]


    def promptEngg(self, task, code, provideDetails=True):
        if task == "Code Completion":
            priming = "Analyze my code and provide synctactically correct and relevant code completion."
            if provideDetails:
                decorator = "Please explain, in detail, how did you complete the code and the underlying rationale behind it."

        elif task == "Code Simplification":
            priming = "Analyze my code and simplify it. If possible, try to reduce the number of lines in the original code using some special features of the detected programming language."
            if provideDetails:
                decorator = "Please explain, in detail, how did you simplify the code and the underlying rationale behind it."

        elif task == "Code Improvisation":
            priming = "Analyze my code and optimize it in terms of time complexity"
            if provideDetails:
                decorator = "Please explain, in detail, how did you optimize the code and the underlying rationale behind it."
        else:
            priming = "Analyze my code and help me debug it."
            if provideDetails:
                decorator = "Please explain, in detail, what you found and why it was a bug."

        if not provideDetails:
            decorator = "After completing your analysis, as instructed to you above, just display your suggestion as code."


        prompt_template ="""
        {priming}

        {code}

        {decorator}
        """

        self.prompt = prompt_template.format(priming=priming,
                                             code=code,
                                             decorator=decorator)


    @retry.Retry()
    def predict(self):
        self.generated_text = LLM.generate_text(prompt = self.prompt,
                                            model = self.LLM_model,
                                            temperature = 0.0).result


# aiPairProgObj = AI_PAIR_PROGRAMMING()





