import nltk

from model.models import UserModelSession, Choice, UserModelRun, Protocol
from model.classifiers import get_emotion, get_sentence_score
import pandas as pd
import numpy as np
import random
from collections import deque
import re
import datetime
import time

nltk.download("wordnet")
from nltk.corpus import wordnet  # noqa


class ModelDecisionMaker:
    def __init__(self):

        # self.kai = pd.read_csv('/Users/yeliu/IC/Individual_Project/code/sat3.0/model/kai.csv',
        #                        encoding='ISO-8859-1')  # change path
        # self.robert = pd.read_csv('/Users/yeliu/IC/Individual_Project/code/sat3.0/model/robert.csv',
        #                           encoding='ISO-8859-1')
        # self.gabrielle = pd.read_csv('/Users/yeliu/IC/Individual_Project/code/sat3.0/model/gabrielle.csv',
        #                              encoding='ISO-8859-1')
        # self.arman = pd.read_csv('/Users/yeliu/IC/Individual_Project/code/sat3.0/model/arman.csv',
        #                          encoding='ISO-8859-1')
        # self.olivia = pd.read_csv('/Users/yeliu/IC/Individual_Project/code/sat3.0/model/olivia.csv',
        #                           encoding='ISO-8859-1')

        # Titles from workshops (Title 7 adapted to give more information)
        self.PROTOCOL_TITLES = [
            "0: None",
            "1: Recalling significant early memories",
            "2: Becoming intimate with your Child",
            "3: Singing a song of affection",
            "4: Expressing love and care for the Child",
            "5: Pledging to care and support our Child",
            "6: Restoring our emotional world after our pledge",
            "7: Maintaining a loving relationship with your Child",
            "8: Creating zest for life",
            "9: Enjoying nature",
            "10: Overcoming your current negative emotions",
            "11: Overcoming past pain",
            "12: Muscle relaxation and playful face",
            "13: Laughing on your own",
            "14: Laughing with your childhood self",  # noqa
            "15: Creating your own brand of laughter",  # noqa
            "16: Learning to change your perspective",
            "17: Learning to be playful about your past pains",
            "18: Identifying our personal resentments and acting them out",  # noqa
            "19: Planning more constructive actions",
            "20: Updating our beliefs to enhance creativity",
            "21: Practicing Affirmations",
        ]

        self.TITLE_TO_PROTOCOL = {
            self.PROTOCOL_TITLES[i]: i for i in range(len(self.PROTOCOL_TITLES))
        }

        self.recent_protocols = deque(maxlen=20)
        self.reordered_protocol_questions = {}
        self.protocols_to_suggest = []

        # Goes from user id to actual value
        self.current_run_ids = {}
        self.current_protocol_ids = {}

        self.current_protocols = {}

        self.positive_protocols = [i for i in range(1, 21)]

        self.INTERNAL_PERSECUTOR_PROTOCOLS = [
            self.PROTOCOL_TITLES[15],
            self.PROTOCOL_TITLES[16],
            self.PROTOCOL_TITLES[8],
            self.PROTOCOL_TITLES[19],
        ]

        # Keys: user ids, values: dictionaries describing each choice (in list)
        # and current choice
        self.user_choices = {}

        # Keys: user ids, values: scores for each question
        # self.user_scores = {}

        # Keys: user ids, values: current suggested protocols
        self.suggestions = {}

        # Tracks current emotion of each user after they classify it
        self.user_emotions = {}

        self.guess_emotion_predictions = {}
        # Structure of dictionary: {question: {
        #                           model_prompt: str or list[str],
        #                           choices: {maps user response to next protocol},
        #                           protocols: {maps user response to protocols to suggest},
        #                           }, ...
        #                           }
        # This could be adapted to be part of a JSON file (would need to address
        # mapping callable functions over for parsing).

        self.users_names = {}
        self.targetA_names = {}
        self.users_feelings = {}
        self.remaining_choices = {}

        self.recent_questions = {}

        self.chosen_personas = {}
        # self.datasets = pd.read_csv('/Users/yeliu/IC/Individual_Project/code/sat3.0/model/sat.csv',
        #                        encoding='ISO-8859-1')
        self.datasets = pd.read_csv('sat.csv')

        self.QUESTIONS = {

            "ask_name": {
                "model_prompt": "Please enter your first name:",
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_name(user_id)
                },
                "protocols": {"open_text": []},
            },

            #  "choose_persona": {
            #     "model_prompt": "Who would you like to talk to?",
            #     "choices": {
            #         "Kai": lambda user_id, db_session, curr_session, app: self.get_kai(user_id),
            #         "Robert": lambda user_id, db_session, curr_session, app: self.get_robert(user_id),
            #         "Gabrielle": lambda user_id, db_session, curr_session, app: self.get_gabrielle(user_id),
            #         "Arman": lambda user_id, db_session, curr_session, app: self.get_arman(user_id),
            #         "Olivia": lambda user_id, db_session, curr_session, app: self.get_olivia(user_id),
            #     },
            #     "protocols": {
            #         "Kai": [],
            #         "Robert": [],
            #         "Gabrielle": [],
            #         "Arman": [],
            #         "Olivia": [],
            #     },
            # },

            "intro_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_intro_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(
                        user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },

            # "opening_prompt": {
            #     "model_prompt": lambda user_id, db_session, curr_session, app: self.get_opening_prompt(user_id),
            #
            #     "choices": {
            #         "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
            #     },
            #     "protocols": {"open_text": []},
            # },

            "guess_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_guess_emotion(
                    user_id, app, db_session
                ),
                "choices": {
                    "Yes": {
                        # "Positive": "after_classification_positive",
                        # "Negative": "after_classification_negative",
                        "Sad": "after_classification_negative",
                        "Angry": "after_classification_negative",
                        "Anxious/Scared": "after_classification_negative",
                        "Happy/Content": "after_classification_positive",
                    },
                    "No": "check_emotion",
                },
                "protocols": {
                    "Yes": [],
                    "No": []
                },
            },

            "check_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_check_emotion(
                    user_id, app, db_session),

                "choices": {
                    # "Positive": lambda user_id, db_session, curr_session, app: self.get_positive_emotion(user_id),
                    # "Negative": lambda user_id, db_session, curr_session, app: self.get_negative_emotion(user_id),
                    "Sad": lambda user_id, db_session, curr_session, app: self.get_sad_emotion(user_id),
                    "Angry": lambda user_id, db_session, curr_session, app: self.get_angry_emotion(user_id),
                    "Anxious/Scared": lambda user_id, db_session, curr_session, app: self.get_anxious_emotion(user_id),
                    "Happy/Content": lambda user_id, db_session, curr_session, app: self.get_happy_emotion(user_id),
                    "Others, but positive feelings": lambda user_id, db_session, curr_session, app: self.get_positive_emotion(user_id),
                    "Others, but negative feelings": lambda user_id, db_session, curr_session,
                                                            app: self.get_negative_emotion(user_id),

                },
                "protocols": {
                    # "Positive": [],
                    # "Negative": []
                    "Sad": [],
                    "Angry": [],
                    "Anxious/Scared": [],
                    "Happy/Content": [],
                    "Others, but positive feelings": [],
                    "Others, but negative feelings": []
                },
            },

            ############# NEGATIVE EMOTIONS (SADNESS, ANGER, FEAR/ANXIETY)

            "after_classification_negative": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_specific_event(
                    user_id, app, db_session),

                "choices": {
                    "Yes, something happened": "event_is_recent",
                    "No, it's just a general feeling": "more_questions",
                },
                "protocols": {
                    "Yes, something happened": [],
                    "No, it's just a general feeling": []
                },
            },

            "event_is_recent": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_event_is_recent(
                    user_id, app, db_session),

                "choices": {
                    "It was recent": "revisiting_recent_events",
                    "It was distant": "revisiting_distant_events",
                },
                "protocols": {
                    "It was recent": [],
                    "It was distant": []
                },
            },

            "revisiting_recent_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_recent(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "more_questions",
                    "No": "more_questions",
                },
                "protocols": {
                    "Yes": [self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[8]],
                    "No": [self.PROTOCOL_TITLES[11]],
                },
            },

            "revisiting_distant_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_distant(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "more_questions",
                    "No": "more_questions",
                },
                "protocols": {
                    "Yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "No": [self.PROTOCOL_TITLES[6]]
                },
            },

            # "project_emotion": {
            #     "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_project_emotion(
            #         user_id, app, db_session),
            #
            #     "choices": {
            #         "Continue": "more_questions",
            #     },
            #     "protocols": {
            #         "Continue": [],
            #     },
            # },

            "more_questions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_more_questions(
                    user_id, app, db_session),

                "choices": {
                    "Continue": "check_targetA",
                    "No, I'd like to see suggestions": "suggestions",
                    "No, I'd like to end the session": "ending_prompt",
                },
                "protocols": {
                    "Continue": [],
                    "No, I'd like to see suggestions": [],
                    "No, I'd like to end the session": [],
                },
            },

            "check_targetA": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_targetA(user_id,
                                                                                                            app,
                                                                                                            db_session),
                "choices": {
                    "Yes": "targetA_finder",
                    "No": "internal_persecutor_accusing",
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "targetA_finder": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_targetA_finder(user_id,
                                                                                                             app,
                                                                                                             db_session),
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_targetA_name(user_id)
                },
                "protocols": {"open_text": []},
            },

            "displaying_antisocial_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_antisocial_emotion(
                    user_id, app, db_session),
                # Envy, jealousy, greed, hatred, mistrust, malevolence, or revengefulness
                "choices": {
                    "Anger": lambda user_id, db_session, curr_session, app: self.get_angry_feeling(user_id),
                    "Envy": lambda user_id, db_session, curr_session, app: self.get_envy_feeling(user_id),
                    "Greed": lambda user_id, db_session, curr_session, app: self.get_greed_feeling(user_id),
                    "Hatred": lambda user_id, db_session, curr_session, app: self.get_hatred_feeling(user_id),
                    "Mistrust": lambda user_id, db_session, curr_session, app: self.get_mistrust_feeling(user_id),
                    "Vengefulness": lambda user_id, db_session, curr_session, app: self.get_vengefulness_feeling(
                        user_id),
                    "Others": "specify_antisocial_emotion",
                    "No, I don't have negative emotions towards them": "check_denial",
                },
                "protocols": {
                    "Anger": [],
                    "Envy": [],
                    "Greed": [],
                    "Hatred": [],
                    "Mistrust": [],
                    "Vengefulness": [],
                    "Others": [],
                    "No, I don't have negative emotions towards them": [],
                },
            },

            "specify_antisocial_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_user_feeling(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_user_feeling(user_id)
                },
                "protocols": {"open_text": []},
            },

            "check_denial": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_denial(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "denial",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question_after_denial(
                        user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "denial": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_denial(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": []
                },
            },

            "displacement": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_displacement(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": []
                },
            },

            "check_A_antisocial_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_A_antisocial_emotion(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "check_fight",
                    "No": "displaying_antisocial_behavior",
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "check_fight": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_fight(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "projective_identification",
                    "No": "projection",
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "projective_identification": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_projective_identification(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            "projection": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_projection(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            "displaying_antisocial_behavior": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_antisocial_behavior(
                    user_id, app, db_session),

                "choices": {
                    "Yes": lambda user_id, db_session, curr_session, app: self.get_next_question_after_takeout(user_id),
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "check_targetB": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_targetB(user_id,
                                                                                                            app,
                                                                                                            db_session),
                "choices": {
                    "Yes": "targetB_finder",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "targetB_finder": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_targetB_finder(user_id,
                                                                                                             app,
                                                                                                             db_session),
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_targetB_name(user_id)
                },
                "protocols": {"open_text": []},
            },

            "check_regression": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_regression(user_id,
                                                                                                               app,
                                                                                                               db_session),
                "choices": {
                    "Yes": "regression",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "regression": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_regression(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            "check_transferance": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_transferance(
                    user_id,
                    app,
                    db_session),
                "choices": {
                    "Yes": "transferance",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "transferance": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_transference(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            "check_reaction_formation": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_reaction_formation(
                    user_id,
                    app,
                    db_session),
                "choices": {
                    "Yes": "reaction_formation",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "reaction_formation": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_reaction_formation(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            # "internal_persecutor_saviour": {
            #     "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_saviour(user_id,
            #                                                                                                  app,
            #                                                                                                  db_session),
            #     "choices": {
            #         "Yes": "project_emotion",
            #         "No": "internal_persecutor_victim",
            #     },
            #     "protocols": {
            #         "Yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
            #         "No": []
            #     },
            # },

            # "internal_persecutor_victim": {
            #     "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_victim(user_id,
            #                                                                                                 app,
            #                                                                                                 db_session),
            #     "choices": {
            #         "Yes": "project_emotion",
            #         "No": "internal_persecutor_controlling",
            #     },
            #     "protocols": {
            #         "Yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
            #         "No": []
            #     },
            # },

            # "internal_persecutor_controlling": {
            #     "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_controlling(
            #         user_id, app, db_session),
            #
            #     "choices": {
            #         "Yes": "project_emotion",
            #         "No": "internal_persecutor_accusing"
            #     },
            #     "protocols": {
            #         "Yes": self.INTERNAL_PERSECUTOR_PROTOCOLS,
            #         "No": []
            #     },
            # },

            "internal_persecutor_accusing": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_accusing(user_id,
                                                                                                              app,
                                                                                                              db_session),

                "choices": {
                    "Yes": "check_projection_internal",
                    "No": "no_mechanism_detected",
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "check_projection_internal": {
                "model_prompt": lambda user_id, db_session, curr_session,
                                       app: self.get_model_prompt_check_projection_internal(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "projection_internal",
                    "No": "check_targetA"
                },
                "protocols": {
                    "Yes": [],
                    "No": []
                },
            },

            "projection_internal": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_projection_internal(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            # "rigid_thought": {
            #     "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_rigid_thought(
            #         user_id, app, db_session),
            #
            #     "choices": {
            #         "Yes": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
            #         "No": "project_emotion",
            #     },
            #     "protocols": {
            #         "Yes": [self.PROTOCOL_TITLES[13]],
            #         "No": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[19]],
            #     },
            # },

            # "personal_crisis": {
            #     "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_personal_crisis(
            #         user_id, app, db_session),
            #
            #     "choices": {
            #         "Yes": "project_emotion",
            #         "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
            #     },
            #     "protocols": {
            #         "Yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
            #         "No": [self.PROTOCOL_TITLES[13]],
            #     },
            # },

            ################# POSITIVE EMOTION (HAPPINESS/CONTENT) #################

            "after_classification_positive": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_happy(user_id, app,
                                                                                                           db_session),

                "choices": {
                    "Continue": "after_classification_negative",
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt"
                },
                "protocols": {
                    "Continue": [],
                    # "Okay": [self.PROTOCOL_TITLES[9], self.PROTOCOL_TITLES[10], self.PROTOCOL_TITLES[11]], #change here?
                    # [self.PROTOCOL_TITLES[k] for k in self.positive_protocols],
                    "See suggestions": [],
                    "End the session": []
                },
            },

            ############################# ALL EMOTIONS #############################

            # "project_emotion": {
            #    "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_project_emotion(user_id, app, db_session),
            #
            #    "choices": {
            #        "Continue": "suggestions",
            #    },
            #    "protocols": {
            #        "Continue": [],
            #    },
            # },

            "no_mechanism_detected": {
                "model_prompt": lambda user_id, db_session, curr_session,
                                       app: self.get_model_prompt_no_mechanism_detected(user_id, app, db_session),

                "choices": {
                    "See suggestions": "suggestions",
                    "Replay": "restart_prompt",  # to-do, restart and data deleted
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "Replay": [],
                    "End the session": [],
                },
            },

            "suggestions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggestions(
                    user_id, app, db_session),

                "choices": {
                    "Yes, I'd love to": "trying_protocol",
                    "No, I'd like to end the session": "ending_prompt",
                    # self.PROTOCOL_TITLES[k]: "trying_protocol"  # self.current_protocol_ids[user_id]
                    # for k in self.positive_protocols
                },
                "protocols": {
                    "Yes, I'd love to": [],
                    "No, I'd like to end the session": [],
                    # self.PROTOCOL_TITLES[k]: [self.PROTOCOL_TITLES[k]]
                    # for k in self.positive_protocols
                },
            },

            "trying_protocol": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_protocol(
                    user_id, app, db_session),

                "choices": {"Continue": "user_found_useful",
                            "End the session": "ending_prompt",
                            },
                "protocols": {"Continue": [], "End the session": []},
            },

            "tip1": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_tip1(
                    user_id, app, db_session),
                "choices": {"Continue": lambda user_id, db_session, curr_session,
                                               app: self.get_model_prompt_tip2(
                    user_id, app, db_session),
                            },
                "protocols": {"Continue": [], },
            },

            "user_found_useful": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(
                    user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better",
                    "I feel worse": "new_protocol_worse",
                    "I feel no change": "new_protocol_same",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },

            "new_protocol_better": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id,
                                                                                                                app,
                                                                                                                db_session),

                "choices": {
                    "Yes , I'd like to see other suggestions": lambda user_id, db_session, curr_session,
                                                               app: self.get_model_prompt_tip1(
                        user_id, app, db_session),
                    # "Yes (replay)": "restart_prompt",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes, I'd like to see other suggestions": [],
                    # "Yes (replay)": [],
                    "No (end session)": []
                },
            },

            "new_protocol_worse": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id,
                                                                                                               app,
                                                                                                               db_session),

                "choices": {
                    "Yes , I'd like to see other suggestions": lambda user_id, db_session, curr_session,
                                                               app: self.get_model_prompt_tip1(
                        user_id, app, db_session),
                    # "Yes (replay)": "restart_prompt",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes , I'd like to see other suggestions": [],
                    # "Yes (replay)": [],
                    "No (end session)": []
                },
            },

            "new_protocol_same": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in your mood.",
                    "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                    "Would you like me to suggest a different protocol?"
                ],

                "choices": {
                    "Yes , I'd like to see other suggestions": lambda user_id, db_session, curr_session,
                                                               app: self.get_model_prompt_tip1(
                        user_id, app, db_session),
                    # "Yes (restart questions)": "restart_prompt",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes , I'd like to see other suggestions": [],
                    # "Yes (restart questions)": [],
                    "No (end session)": []
                },
            },

            "ending_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ending(user_id,
                                                                                                            app,
                                                                                                            db_session),

                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []}
            },

            "restart_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_restart_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(
                        user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },
        }
        self.QUESTION_KEYS = list(self.QUESTIONS.keys())

    def initialise_prev_questions(self, user_id):
        self.recent_questions[user_id] = []

    def clear_persona(self, user_id):
        self.chosen_personas[user_id] = ""

    def clear_names(self, user_id):
        self.users_names[user_id] = ""

    def clear_datasets(self, user_id):
        self.datasets[user_id] = pd.DataFrame(columns=['sentences'])

    def initialise_remaining_choices(self, user_id):
        self.remaining_choices[user_id] = ["check_reaction_formation", "check_targetB",
                                           "check_transferance", "check_regression"]

    def save_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["ask_name"]
        except:  # noqa
            user_response = ""
        self.users_names[user_id] = user_response
        return "intro_prompt"

    def save_targetA_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["targetA_finder"].lstrip().rstrip()
        except:  # noqa
            user_response = ""
        self.targetA_names[user_id] = user_response if user_response != "" else "X"
        return "displaying_antisocial_emotion"

    def save_targetB_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["targetB_finder"].lstrip().rstrip()
        except:  # noqa
            user_response = ""
        self.targetA_names[user_id] = user_response if user_response != "" else "Y"
        return "displacement"

    def save_user_feeling(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["specify_antisocial_emotion"]
        except:  # noqa
            user_response = "general negative feeling"
        self.users_feelings[user_id] = user_response
        return "check_A_antisocial_emotion"

    def get_suggestions(self, user_id,
                        app):  # from all the lists of protocols collected at each step of the dialogue it puts together some and returns these as suggestions
        suggestions = []
        for curr_suggestions in list(self.suggestions[user_id]):
            if len(curr_suggestions) > 2:
                i, j = random.choices(range(0, len(curr_suggestions)), k=2)
                if curr_suggestions[i] and curr_suggestions[
                    j] in self.PROTOCOL_TITLES:  # weeds out some gibberish that im not sure why it's there
                    suggestions.extend([curr_suggestions[i], curr_suggestions[j]])
            else:
                suggestions.extend(curr_suggestions)
            suggestions = set(suggestions)
            suggestions = list(suggestions)
        while len(suggestions) < 4:  # augment the suggestions if less than 4, we add random ones avoiding repetitions
            p = random.choice([i for i in range(1, 20) if
                               i not in [6, 11]])  # we dont want to suggest protocol 6 or 11 at random here
            if (any(self.PROTOCOL_TITLES[p] not in curr_suggestions for curr_suggestions in
                    list(self.suggestions[user_id]))
                    and self.PROTOCOL_TITLES[p] not in self.recent_protocols and self.PROTOCOL_TITLES[
                        p] not in suggestions):
                suggestions.append(self.PROTOCOL_TITLES[p])
                self.suggestions[user_id].extend([self.PROTOCOL_TITLES[p]])
        return suggestions

    def clear_suggestions(self, user_id):
        self.suggestions[user_id] = []
        self.reordered_protocol_questions[user_id] = deque(maxlen=5)

    def clear_emotion_scores(self, user_id):
        self.guess_emotion_predictions[user_id] = ""

    def create_new_run(self, user_id, db_session, user_session):
        new_run = UserModelRun(session_id=user_session.id)
        db_session.add(new_run)
        db_session.commit()
        self.current_run_ids[user_id] = new_run.id
        return new_run

    def clear_choices(self, user_id):
        self.user_choices[user_id] = {}

    def update_suggestions(self, user_id, protocols, app):

        # Check if user_id already has suggestions
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []

        if type(protocols) != list:
            self.suggestions[user_id].append(deque([protocols]))
        else:
            self.suggestions[user_id].append(deque(protocols))

    # Takes next item in queue, or moves on to suggestions
    # if all have been checked

    # def get_kai(self, user_id):
    #    self.chosen_personas[user_id] = "Kai"
    #    self.datasets[user_id] = self.kai
    #    return "opening_prompt"
    # def get_robert(self, user_id):
    #    self.chosen_personas[user_id] = "Robert"
    #    self.datasets[user_id] = self.robert
    #    return "opening_prompt"
    # def get_gabrielle(self, user_id):
    #    self.chosen_personas[user_id] = "Gabrielle"
    #    self.datasets[user_id] = self.gabrielle
    #    return "opening_prompt"
    # def get_arman(self, user_id):
    #    self.chosen_personas[user_id] = "Arman"
    #    self.datasets[user_id] = self.arman
    #    return "opening_prompt"
    # def get_olivia(self, user_id):
    #    self.chosen_personas[user_id] = "Olivia"
    #    self.datasets[user_id] = self.olivia
    #    return "opening_prompt"

    def get_intro_prompt(self, user_id):
        name = self.users_names[user_id] if self.users_names[user_id] != "" else "there"
        welcome = "Hi " + name + "! Nice to meet you!"
        intro_prompt0 = "My name is Kai, your emotional support assistant."
        intro_prompt1 = "I'm here to help you to understand some of the defense mechanisms that you may unconsciously employ when you encounter negative emotions."
        intro_prompt2 = "To start, please could you tell me how you are feeling today?"
        # if self.users_names[user_id] == "":
        #     opening_prompt = ["Hello, this is " + self.chosen_personas[user_id] + ". ", "How are you feeling today?"]
        # else:
        #     opening_prompt = ["Hello " + self.users_names[user_id] + ", this is " + self.chosen_personas[user_id] + ". ", "How are you feeling today?"]
        return welcome, intro_prompt0, intro_prompt1, intro_prompt2

    # def get_opening_prompt(self, user_id):
    #     if self.users_names[user_id] == "":
    #         opening_prompt = ["Hello, this is " + self.chosen_personas[user_id] + ". ", "How are you feeling today?"]
    #     else:
    #         opening_prompt = [
    #             "Hello " + self.users_names[user_id] + ", this is " + self.chosen_personas[user_id] + ". ",
    #             "How are you feeling today?"]
    #     return opening_prompt

    def get_restart_prompt(self, user_id):
        name = self.users_names[user_id] if self.users_names[user_id] != "" else "my friend"
        # if self.users_names[user_id] == "":
        #     restart_prompt = ["Please tell me again, how are you feeling today?"]
        # else:
        #     restart_prompt = ["Please could you tell me again, " + self.users_names[user_id] + ", how are you feeling today?"]
        restart_prompt = [
            "Please could you tell me again, " + self.users_names[user_id] + ", how are you feeling today?"]
        return restart_prompt

    def get_next_question(self, user_id):
        if self.remaining_choices[user_id] == []:
            return "no_mechanism_detected"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice

    def get_next_question_after_takeout(self, user_id):
        self.remaining_choices[user_id].remove("check_targetB")  # delete check_targetB from remaining_choices

        if self.remaining_choices[user_id] == []:
            return "no_mechanism_detected"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice

    def get_next_question_after_denial(self, user_id):
        self.remaining_choices[user_id].remove("check_targetB")
        self.remaining_choices[user_id].remove("check_reaction_formation") # delete check_targetB from remaining_choices

        if self.remaining_choices[user_id] == []:
            return "no_mechanism_detected"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice

    def get_denial(self, user_id):
        question = "Thank you, I really appreciate your cooperation. It looks like you are using denial mechanism. Denial usually happens when you refuse to accept facts or reality, thus block external events from awareness."
        return self.split_sentence(question)

    def get_displacement(self, user_id):
        question = "Thank you, I really appreciate your time and patience. From what you've told me, it is possible that you are using displacement mechanism. This happens when you redirect your negative emotion from its original source to a less threatening recipient."
        return self.split_sentence(question), "In your case, it looks like you redirect your {} from {} to {}.".format(
            self.users_feelings[user_id], self.targetA_names[user_id], self.targetB_names[user_id])

    def get_transference(self, user_id):
        question = "Great job, you've done so well with this. Thank you very much for your time and efforts. It seems like you are using transference mechanism. This usually happens when you redirect some of your feelings about one person (usually your primary caregiver) to an entirely different person."
        return self.split_sentence(
            question), "In your case, it looks like you transfer your feelings from your primary caregiver to {} due to their similarities.".format(
            self.users_feelings[user_id], self.targetA_names[user_id])

    def get_regression(self, user_id):
        question = "Thank you very much, I really appreciate your cooperation. I think you are probably using regression mechanism. It is a way to protect yourself in stressful events by going back in time to a period when you safe, instead of handling the unacceptable facts in a more adult manner."
        return self.split_sentence(question)

    def get_projection(self, user_id):
        question = "Great job, I really appreciate your cooperation. From what you've told me, you are probably using projection mechanism. Projection happens when you recognize your unacceptable thoughts in someone else to avoid recognizing those thoughts in yourself subconsciously."
        return self.split_sentence(question), "In your case, it's likely you attribute your {} to {}.".format(
            self.users_feelings[user_id], self.targetA_names[user_id])

    def get_projection_internal(self, user_id):
        question = "Thank you very much! From what you have told me, it's likely that you are using projection mechanism. It's possible you project the internal persecutor to yourself and thus develop an inner persecutor as a way to protect yourself."
        # return "Thank you, I really appreciate your cooperation. It looks like you are using projection mechanism. This usually happens when you recognize your unacceptable traits or impulses in someone else to avoid recognizing those traits or impulses in yourself subconsciously."
        return self.split_sentence(question)

    def get_reaction_formation(self, user_id):
        question = "Thanks a lot, I really appreciate your cooperation. It's possible that you are using reaction formation mechanism. This usually happens when you unconsciously behaves in the opposite way to which you think or feel, in order to hide your true feelings."
        return self.split_sentence(
            question), "In your case, it looks like you treat {} in a very friendly way to hide your {}.".format(
            self.targetA_names[user_id], self.users_feelings[user_id])

    def get_projective_identification(self, user_id):
        question = "Well done, thank you for your trust and cooperation! It looks like you are using projective identification mechanism. This usually happens when you project unacceptable thoughts onto another person, and that person introjects the projected thoughts and believes themselves to be characterized by you appropriately and justifiably."
        return self.split_sentence(
            question), "In your case, it's likely you attribute your {} to {}, and somehow {} accepts and resonates this projection, which may cause a fight as a result.".format(
            self.users_feelings[user_id], self.targetA_names[user_id], self.targetA_names[user_id])

    def add_to_reordered_protocols(self, user_id, next_protocol):
        self.reordered_protocol_questions[user_id].append(next_protocol)

    def add_to_next_protocols(self, next_protocols):
        self.protocols_to_suggest.append(deque(next_protocols))

    def clear_suggested_protocols(self):
        self.protocols_to_suggest = []

    # NOTE: this is not currently used, but can be integrated to support
    # positive protocol suggestions (to avoid recent protocols).
    # You would need to add it in when a user's emotion is positive
    # and they have chosen a protocol.

    def add_to_recent_protocols(self, recent_protocol):
        if len(self.recent_protocols) == self.recent_protocols.maxlen:
            # Removes oldest protocol
            self.recent_protocols.popleft()
        self.recent_protocols.append(recent_protocol)

    def determine_next_prompt_opening(self, user_id, app, db_session):
        user_response = self.user_choices[user_id]["choices_made"]["intro_prompt"]
        emotion = get_emotion(user_response)
        if emotion == 'fear':
            self.guess_emotion_predictions[user_id] = 'fear'
            self.user_emotions[user_id] = 'fear'
        elif emotion == 'sadness':
            self.guess_emotion_predictions[user_id] = 'sad'
            self.user_emotions[user_id] = 'sad'
        elif emotion == 'anger':
            self.guess_emotion_predictions[user_id] = 'angry'
            self.user_emotions[user_id] = 'angry'
        else:
            self.guess_emotion_predictions[user_id] = 'happy'
            self.user_emotions[user_id] = 'happy'
        # self.guess_emotion_predictions[user_id] = emotion
        # self.user_emotions[user_id] = emotion
        return "guess_emotion"

    def get_best_sentence(self, column, prev_qs):
        # return random.choice(column.dropna().sample(n=15).to_list()) #using random choice instead of machine learning
        maxscore = 0
        chosen = ''
        for row in column.dropna().sample(n=5):  # was 25
            fitscore = get_sentence_score(row, prev_qs)
            if fitscore > maxscore:
                maxscore = fitscore
                chosen = row
        if chosen != '':
            return chosen
        else:
            return random.choice(column.dropna().sample(n=5).to_list())  # was 25

    def split_sentence(self, sentence):
        temp_list = re.split('(?<=[.?!]) +', sentence)
        if '' in temp_list:
            temp_list.remove('')
        temp_list = [i + " " if i[-1] in [".", "?", "!"] else i for i in temp_list]
        if len(temp_list) == 2:
            return temp_list[0], temp_list[1]
        elif len(temp_list) == 3:
            return temp_list[0], temp_list[1], temp_list[2]
        elif len(temp_list) == 4:
            return temp_list[0], temp_list[1], temp_list[2], temp_list[3]
        elif len(temp_list) == 5:
            return temp_list[0], temp_list[1], temp_list[2], temp_list[3], temp_list[4]
        elif len(temp_list) == 6:
            return temp_list[0], temp_list[1], temp_list[2], temp_list[3], temp_list[4], temp_list[5]
        elif len(temp_list) == 7:
            return temp_list[0], temp_list[1], temp_list[2], temp_list[3], temp_list[4], temp_list[5], temp_list[6]
        else:
            return sentence

    def get_model_prompt_guess_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["From what you have said I believe you are feeling {}. Is this correct?"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.guess_emotion_predictions[user_id].lower())
        return self.split_sentence(question)
        # emotion = self.guess_emotion_predictions[user_id]
        # return "From what you have said I believe you are feeling " + emotion + ". Is this correct?"

    def get_model_prompt_check_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data[
            "I am sorry. Please select from the emotions below the one that best reflects what you are feeling:"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
        # return "I am sorry. Please select from the emotions below the one that best reflects what you are feeling:"

    def get_positive_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "positive"
        self.user_emotions[user_id] = "positive"
        return "after_classification_positive"

    def get_negative_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "negative"
        self.user_emotions[user_id] = "negative"
        return "after_classification_negative"

    def get_angry_feeling(self, user_id):
        self.users_feelings[user_id] = "angry"
        return "check_A_antisocial_emotion"

    def get_envy_feeling(self, user_id):
        self.users_feelings[user_id] = "envy"
        return "check_A_antisocial_emotion"

    def get_greed_feeling(self, user_id):
        self.users_feelings[user_id] = "greed"
        return "check_A_antisocial_emotion"

    def get_hatred_feeling(self, user_id):
        self.users_feelings[user_id] = "hatred"
        return "check_A_antisocial_emotion"

    def get_mistrust_feeling(self, user_id):
        self.users_feelings[user_id] = "mistrust"
        return "check_A_antisocial_emotion"

    def get_vengefulness_feeling(self, user_id):
        self.users_feelings[user_id] = "vengefulness"
        return "check_A_antisocial_emotion"

    def get_user_feeling(self, user_id):
        question = "Alright, thanks. Could you please specify your childhood-self's negative feelings towards " + \
                   self.targetA_names[
                       user_id] + "? If you don't know what specific feeling your childhood-self has, please input 'general negative feeling' or just send me a blank message."
        return self.split_sentence(question)

    def get_sad_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "sad"
        self.user_emotions[user_id] = "sad"
        return "after_classification_negative"

    def get_angry_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "angry"
        self.user_emotions[user_id] = "angry"
        return "after_classification_negative"

    def get_anxious_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "fear"
        self.user_emotions[user_id] = "fear"
        return "after_classification_negative"

    def get_happy_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "happy"
        self.user_emotions[user_id] = "happy"
        return "after_classification_positive"

    def get_model_prompt_project_emotion(self, user_id, app, db_session):
        # if self.chosen_personas[user_id] == "Robert":
        #     prompt = "Ok, thank you. Now, one last important thing: since you've told me you're feeling " + self.user_emotions[user_id].lower() + ", I would like you to try to project this emotion onto your childhood self. You can press 'continue' when you are ready and I'll suggest some protocols I think may be appropriate for you."
        # elif self.chosen_personas[user_id] == "Gabrielle":
        #     prompt = "Thank you, I will recommend some protocols for you in a moment. Before I do that, could you please try to project your " + self.user_emotions[user_id].lower() + " feeling onto your childhood self? Take your time to try this, and press 'continue' when you feel ready."
        # elif self.chosen_personas[user_id] == "Arman":
        #     prompt = "Ok, thank you for letting me know that. Before I give you some protocol suggestions, please take some time to project your current " + self.user_emotions[user_id].lower() + " feeling onto your childhood self. Press 'continue' when you feel able to do it."
        # elif self.chosen_personas[user_id] == "Arman":
        #     prompt = "Ok, thank you, I'm going to draw up a list of protocols which I think would be suitable for you today. In the meantime, going back to this " + self.user_emotions[user_id].lower() + " feeling of yours, would you like to try to project it onto your childhood self? You can try now and press 'continue' when you feel ready."
        # else:
        #     prompt = "Thank you. While I have a think about which protocols would be best for you, please take your time now and try to project your current " + self.user_emotions[user_id].lower() + " emotion onto your childhood self. When you are able to do this, please press 'continue' to receive your suggestions."
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        # data = self.datasets
        # base_prompt = "Project onto childhood-self"
        # column = data[base_prompt].dropna()
        # question = self.get_best_sentence(column, prev_qs)
        # if len(self.recent_questions[user_id]) < 50:
        #     self.recent_questions[user_id].append(question)
        # else:
        #     self.recent_questions[user_id] = []
        #     self.recent_questions[user_id].append(question)
        # return self.split_sentence(question)
        question = "Thank you, I will help you deal with your emotions in a moment. Before I do that, could you please try to project your negatiive feeling onto your childhood self? Take your time to try this, and press 'Continue' when you feel ready. And I will ask you some questions about your childhood-self to get a better understanding of your situation."
        return self.split_sentence(question)

    # def get_model_prompt_project_emotion(self, user_id, app, db_session):
    #     time.sleep(7)
    #     if self.chosen_personas[user_id] == "Robert":
    #         prompt = "Ok, thank you. Now, one last important thing: since you've told me you're feeling " + self.user_emotions[user_id].lower() + ", I would like you to try to project this emotion onto your childhood self. You can press 'continue' when you are ready and I'll suggest some protocols I think may be appropriate for you."
    #     elif self.chosen_personas[user_id] == "Gabrielle":
    #         prompt = "Thank you, I will recommend some protocols for you in a moment. Before I do that, could you please try to project your " + self.user_emotions[user_id].lower() + " feeling onto your childhood self? Take your time to try this, and press 'continue' when you feel ready."
    #     elif self.chosen_personas[user_id] == "Arman":
    #         prompt = "Ok, thank you for letting me know that. Before I give you some protocol suggestions, please take some time to project your current " + self.user_emotions[user_id].lower() + " feeling onto your childhood self. Press 'continue' when you feel able to do it."
    #     elif self.chosen_personas[user_id] == "Arman":
    #         prompt = "Ok, thank you, I'm going to draw up a list of protocols which I think would be suitable for you today. In the meantime, going back to this " + self.user_emotions[user_id].lower() + " feeling of yours, would you like to try to project it onto your childhood self? You can try now and press 'continue' when you feel ready."
    #     else:
    #         prompt = "Thank you. While I have a think about which protocols would be best for you, please take your time now and try to project your current " + self.user_emotions[user_id].lower() + " emotion onto your childhood self. When you are able to do this, please press 'continue' to receive your suggestions."
    #     return self.split_sentence(prompt)

    # def get_model_prompt_saviour(self, user_id, app, db_session):
    #     prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
    #     data = self.datasets
    #     base_prompt = self.user_emotions[user_id] + " - Do you believe that you should be the saviour of someone else?"
    #     column = data[base_prompt].dropna()
    #     question = self.get_best_sentence(column, prev_qs)
    #     if len(self.recent_questions[user_id]) < 50:
    #         self.recent_questions[user_id].append(question)
    #     else:
    #         self.recent_questions[user_id] = []
    #         self.recent_questions[user_id].append(question)
    #     return self.split_sentence(question)

    # def get_model_prompt_victim(self, user_id, app, db_session):
    #     prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
    #     data = self.datasets
    #     base_prompt = self.user_emotions[
    #                       user_id] + " - Do you see yourself as the victim, blaming someone else for how negative you feel?"
    #     column = data[base_prompt].dropna()
    #     question = self.get_best_sentence(column, prev_qs)
    #     if len(self.recent_questions[user_id]) < 50:
    #         self.recent_questions[user_id].append(question)
    #     else:
    #         self.recent_questions[user_id] = []
    #         self.recent_questions[user_id].append(question)
    #     return self.split_sentence(question)

    # def get_model_prompt_controlling(self, user_id, app, db_session):
    #     prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
    #     data = self.datasets
    #     base_prompt = self.user_emotions[user_id] + " - Do you feel that you are trying to control someone?"
    #     column = data[base_prompt].dropna()
    #     question = self.get_best_sentence(column, prev_qs)
    #     if len(self.recent_questions[user_id]) < 50:
    #         self.recent_questions[user_id].append(question)
    #     else:
    #         self.recent_questions[user_id] = []
    #         self.recent_questions[user_id].append(question)
    #     return self.split_sentence(question)

    def get_model_prompt_accusing(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Are you always blaming and accusing yourself for when something goes wrong?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "Are you always blaming and accusing yourself for when something goes wrong?"

    def get_model_prompt_specific_event(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "specific event"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "Was this caused by a specific event/s?"

    def get_model_prompt_event_is_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Was this caused by a recent or distant event (or events)?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "Was this caused by a recent or distant event (or events)?"

    def get_model_prompt_revisit_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Have you recently attempted protocol 9 and found this reignited unmanageable emotions as a result of old events?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "Have you recently attempted protocol 11 and found this reignited unmanageable emotions as a result of old events?"

    def get_model_prompt_revisit_distant(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Have you recently attempted protocol 10 and found this reignited unmanageable emotions as a result of old events?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "Have you recently attempted protocol 6 and found this reignited unmanageable emotions as a result of old events?"

    def get_model_prompt_more_questions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Thank you. Now I will ask some questions to understand your situation."
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        question1 = "Before I do that, could you please try to project your negative feeling onto your childhood self?"
        question2 = "Take your time to try this, and press 'Continue' when you feel ready."
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question), question1, question2
        # return "Thank you. Now I will ask some questions to understand your situation."

    def get_model_check_targetA(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["check targeta"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
        # return "Good job! Then I was wondering if this negative emotion has something to do with your interaction with someone else?"

    def get_model_check_targetB(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["check targetb"].dropna()  # to-do delete name request
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.users_feelings[user_id], self.targetA_names[user_id])
        return self.split_sentence(question)
        # return "I understand, thank you for letting me know. So you find that your childhood-self has felt this negative emotion towards" + \
        #        self.targetA_names[
        #            user_id] + "but kept it to themselves. Now I was wondering if your childhood-self is releasing this pent up emotions towards someone else?"

    def get_model_targetA_finder(self, user_id, app, db_session):
        question = "Okay, thank you for your response. Could you please tell me the name of this person if it's convenient? You may send me a blank messgae or input 'X' if you are not comfortable sharing the name, and I fully understand your concern."
        return self.split_sentence(question)

    def get_model_targetB_finder(self, user_id, app, db_session):
        question = "May I have the name of this person please, if you don't mind? You may send me a blank messgae or input 'Y' if you are not comfortable sharing their names, I completely understand your concern."
        return self.split_sentence(question)

    def get_model_check_denial(self, user_id, app, db_session):
        question = "You are doing a great job so far! If it's ok to proceed, may I ask you to think one step further? Would you agree it's possible that you actually have some negative feeling towards " + \
                   self.targetA_names[
                       user_id] + ", but somehow you are not fully aware of it or choose to deny it? I'm sorry that my words may sound a little bit intrusive, but I am trying to understand your emotion better and hope you understand that denial is common and normal, especially in times of great stress."
        return self.split_sentence(question)

    def get_model_prompt_check_projection_internal(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["childhood trauma"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)
        # return "I understand that thinking about difficult events that have only just happened can be uncomfortable at times. Has your childhood-self ever experienced some serious illness/ trauma, when people abused you physically, sexually, psychologically?"

    def get_model_check_regression(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["childlike"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        return self.split_sentence(question)
        # return "Has your childhood-self behaviors become more childlike or primitive with respect to " + \
        #        self.targetA_names[user_id] + " ?"

    def get_model_check_transferance(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["caregiver"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        question += " Especially when " + self.targetA_names[
            user_id] + " and their primary caregiver look alike or have similar personalities."
        return self.split_sentence(question)
        # return "Is the feelings/attitudes your childhood-self have about " + self.targetA_names[
        #     user_id] + " similar to any of their primary caregivers (such as parents or relatives who take the most of the responsibility of taking care of you during childhood)?"

    def get_model_check_reaction_formation(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["friendly"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        return self.split_sentence(question)
        # return "Has your childhood-self treated " + self.targetA_names[user_id] + " in an excessively friendly manner?"

    def get_model_antisocial_emotion(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        # data = self.datasets
        # base_prompt = self.user_emotions[user_id] + " - Have you strongly felt or expressed any of the following emotions towards someone:"
        # column = data[base_prompt].dropna()
        # question = self.get_best_sentence(column, prev_qs)
        # if len(self.recent_questions[user_id]) < 50:
        #     self.recent_questions[user_id].append(question)
        # else:
        #     self.recent_questions[user_id] = []
        #     self.recent_questions[user_id].append(question)
        # return [self.split_sentence(question), "Envy, jealousy, greed, hatred, mistrust, malevolence, or revengefulness?"]
        question = "Thank you. We all feel emotions and sometimes express them towards other people, and sometimes these emotions can be quite strong. Could you tell me whether your childhood-self has been feeling any of the following negative emotions towards " + \
                   self.targetA_names[
                       user_id] + "?"
        return self.split_sentence(question)

    def get_model_A_antisocial_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["TargetA attitude"].dropna()
        my_string = self.get_best_sentence(column, prev_qs) + " Particularly, " + self.users_feelings[
            user_id] + ", like your childhood-self has towards " + self.targetA_names[user_id] + " ?"
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        return self.split_sentence(question)
        # return "Do you think " + self.targetA_names[user_id] + " also felt " + self.users_feelings[
        #     user_id] + " towards your childhood-self?"

    def get_model_check_fight(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["fight"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        return self.split_sentence(question)
        # return "Has your childhood-self had a fight with " + self.targetA_names[user_id] + " ?"

    def get_model_antisocial_behavior(self, user_id, app, db_session):
        question = "I see, thank you for taking the time to answer all these questions, I am still trying to figure out the best way to help. I wonder if your childhood-self has expressed their " + \
                   self.users_feelings[user_id] + " towards " + self.targetA_names[
                       user_id] + "?"
        return self.split_sentence(question)

    def get_model_prompt_no_mechanism_detected(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["Fail to detect"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "I am sorry that I cannot detect any defense mechanism. Please press 'Tips' to get some tips on how to cope with negative feelings that you might find useful, press 'Replay' to play it again or press 'End' to end the conversation."

    # def get_model_prompt_rigid_thought(self, user_id, app, db_session):
    #     prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
    #     data = self.datasets
    #     base_prompt = self.user_emotions[
    #                       user_id] + " - In previous conversations, have you considered other viewpoints presented?"
    #     column = data[base_prompt].dropna()
    #     question = self.get_best_sentence(column, prev_qs)
    #     if len(self.recent_questions[user_id]) < 50:
    #         self.recent_questions[user_id].append(question)
    #     else:
    #         self.recent_questions[user_id] = []
    #         self.recent_questions[user_id].append(question)
    #     return self.split_sentence(question)

    # def get_model_prompt_personal_crisis(self, user_id, app, db_session):
    #     prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
    #     data = self.datasets
    #     base_prompt = self.user_emotions[
    #                       user_id] + " - Are you undergoing a personal crisis (experiencing difficulties with loved ones e.g. falling out with friends)?"
    #     column = data[base_prompt].dropna()
    #     question = self.get_best_sentence(column, prev_qs)
    #     if len(self.recent_questions[user_id]) < 50:
    #         self.recent_questions[user_id].append(question)
    #     else:
    #         self.recent_questions[user_id] = []
    #         self.recent_questions[user_id].append(question)
    #     return self.split_sentence(question)

    def get_model_prompt_happy(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["Recall painful event"].dropna()
        question = self.get_best_sentence(column,
                                          prev_qs) + " Please press 'Continue' when you are ready. Or if you are not in the mood, I also have some suggestions on how to handle difficult emotions if you are interested by any chance. Feel free to end the conversation whenever you feel uncomfortable."
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "That's good! I am happy to hear that you are feeling content now. But may I ask you to recall a painful episode when you had difficult emotions in the past? Feel free to quit if you are not in the mood. Press 'Continue' when you are ready."

    def get_model_prompt_suggestions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["Here are my tips"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # question = "It is not easy to overcome difficult emotions. For more help, I've put together some tips on how to cope with negative feelings that you might find useful. Are you interested in trying some of them by any chance?"
        # return self.split_sentence(question)
        # return "It is not easy to overcome difficult emotions. For more help, I've put together some tips on how to cope with negative feelings that you might find useful. Are you interested in trying some of them by any chance?"


    def get_model_prompt_trying_protocol(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        # data = self.datasets
        # column = data[
        #     "All emotions - Please try to go through this protocol now. When you finish, press 'continue'"].dropna()
        # question = self.get_best_sentence(column, prev_qs)
        # if len(self.recent_questions[user_id]) < 50:
        #     self.recent_questions[user_id].append(question)
        # else:
        #     self.recent_questions[user_id] = []
        #     self.recent_questions[user_id].append(question)
        # return ["You have selected Protocol " + str(self.current_protocol_ids[user_id][0]) + ". ",
        #         self.split_sentence(question)]
        question = "First of all, it is essential to recognize your negative emotions and act them out in a mature way. Actually, there is a defense mechanism called 'sublimation', which means to displace your unacceptable emotions into constructive and socially acceptable behaviors. Maybe you can try to practice protocl 8, 17 or 18,  I really hope the protocols will make you feel better. Please press 'Continue' when you finish."
        return self.split_sentence(question)

    def get_model_prompt_tip1(self, user_id, app, db_session):
        question = "Adopting a thir-person perspective is helpful to reduce the intensity of your negative emotions. Hence, why not try protocol 15 and 19 when you get stuck in negative emotion. Please press 'Continue' if you would like to know more."
        return self.split_sentence(question)

    def get_model_prompt_tip2(self, user_id, app, db_session):
        question = "Ultimately, laughter is the best medicine. Digesting negative emotions into laughter will make you feel better. Try protocol 11, 12, 13, 14 and 16, and learn to laugh at your failures or disturbing events."
        return self.split_sentence(question)

    def get_model_prompt_found_useful(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["All emotions - Do you feel better or worse after having taken this protocol?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "Have you found this suggestion useful?"

    def get_model_prompt_new_better(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        print(data.columns)
        column = data["Patient feels better"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "Would you like to attempt another protocol?"

    def get_model_prompt_new_worse(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["Patient feels worse"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)
        # return "Would you like to attempt another protocol?"

    def get_model_prompt_ending(self, user_id, app, db_session):
        # prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        # data = self.datasets
        # column = data["All emotions - Thank you for taking part. See you soon"].dropna()
        # question = self.get_best_sentence(column, prev_qs)
        # if len(self.recent_questions[user_id]) < 50:
        #     self.recent_questions[user_id].append(question)
        # else:
        #     self.recent_questions[user_id] = []
        #     self.recent_questions[user_id].append(question)
        # return [self.split_sentence(question),
        #         "You have been disconnected. Refresh the page if you would like to start over."]
        question = "Thank you for taking part. I appreciate that you've spent this time with me. I look forward to seeing you next time. You have been disconnected. Refresh the page if you would like to start over."
        return self.split_sentence(question)

    def determine_next_prompt_new_protocol(self, user_id, app):
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []
        if len(self.suggestions[user_id]) > 0:
            return "suggestions"
        return "more_questions"

    # def determine_positive_protocols(self, user_id, app):
    #     protocol_counts = {}
    #     total_count = 0
    #
    #     for protocol in self.positive_protocols:
    #         count = Protocol.query.filter_by(protocol_chosen=protocol).count()
    #         protocol_counts[protocol] = count
    #         total_count += count
    #
    #     # for protocol in counts:
    #     if total_count > 10:
    #         first_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
    #         del protocol_counts[first_item]
    #
    #         second_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
    #         del protocol_counts[second_item]
    #
    #         third_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
    #         del protocol_counts[third_item]
    #     else:
    #         # CASE: < 10 protocols undertaken in total, so randomness introduced
    #         # to avoid lowest 3 being recommended repeatedly.
    #         # Gives number of next protocol to be suggested
    #         first_item = np.random.choice(
    #             list(set(self.positive_protocols) - set(self.recent_protocols))
    #         )
    #         second_item = np.random.choice(
    #             list(
    #                 set(self.positive_protocols)
    #                 - set(self.recent_protocols)
    #                 - set([first_item])
    #             )
    #         )
    #         third_item = np.random.choice(
    #             list(
    #                 set(self.positive_protocols)
    #                 - set(self.recent_protocols)
    #                 - set([first_item, second_item])
    #             )
    #         )
    #
    #     return [
    #         self.PROTOCOL_TITLES[first_item],
    #         self.PROTOCOL_TITLES[second_item],
    #         self.PROTOCOL_TITLES[third_item],
    #     ]

    # def determine_protocols_keyword_classifiers(
    #         self, user_id, db_session, curr_session, app
    # ):
    #
    #     # We add "suggestions" first, and in the event there are any left over we use those, otherwise we divert past it.
    #     self.add_to_reordered_protocols(user_id, "suggestions")
    #
    #     # Default case: user should review protocols 13 and 14.
    #     # self.add_to_next_protocols([self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]])
    #     return self.get_next_protocol_question(user_id, app)

    def update_conversation(self, user_id, new_dialogue, db_session, app):
        try:
            session_id = self.user_choices[user_id]["current_session_id"]
            curr_session = UserModelSession.query.filter_by(id=session_id).first()
            if curr_session.conversation is None:
                curr_session.conversation = "" + new_dialogue
            else:
                curr_session.conversation = curr_session.conversation + new_dialogue
            curr_session.last_updated = datetime.datetime.utcnow()
            db_session.commit()
        except KeyError:
            curr_session = UserModelSession(
                user_id=user_id,
                conversation=new_dialogue,
                last_updated=datetime.datetime.utcnow(),
            )

            db_session.add(curr_session)
            db_session.commit()
            self.user_choices[user_id]["current_session_id"] = curr_session.id

    def save_current_choice(
            self, user_id, input_type, user_choice, user_session, db_session, app
    ):
        # Set up dictionary if not set up already
        # with Session() as session:

        try:
            self.user_choices[user_id]
        except KeyError:
            self.user_choices[user_id] = {}

        # Define default choice if not already set
        try:
            current_choice = self.user_choices[user_id]["choices_made"][
                "current_choice"
            ]
        except KeyError:
            current_choice = self.QUESTION_KEYS[0]

        try:
            self.user_choices[user_id]["choices_made"]
        except KeyError:
            self.user_choices[user_id]["choices_made"] = {}

        if current_choice == "ask_name":
            self.clear_suggestions(user_id)
            self.user_choices[user_id]["choices_made"] = {}
            self.create_new_run(user_id, db_session, user_session)

        # Save current choice
        self.user_choices[user_id]["choices_made"]["current_choice"] = current_choice
        self.user_choices[user_id]["choices_made"][current_choice] = user_choice

        curr_prompt = self.QUESTIONS[current_choice]["model_prompt"]
        # prompt_to_use = curr_prompt
        if callable(curr_prompt):
            curr_prompt = curr_prompt(user_id, db_session, user_session, app)

        # removed stuff here

        else:
            self.update_conversation(
                user_id,
                "Model:{} \nUser:{} \n".format(curr_prompt, user_choice),
                db_session,
                app,
            )

        # Case: update suggestions for next attempt by removing relevant one
        # if (
        #         current_choice == "suggestions"
        # ):
        #
        #     # PRE: user_choice is a string representing a number from 1-20,
        #     # or the title for the corresponding protocol
        #
        #     try:
        #         current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
        #     except KeyError:
        #         current_protocol = int(user_choice)
        #
        #     protocol_chosen = Protocol(
        #         protocol_chosen=current_protocol,
        #         user_id=user_id,
        #         session_id=user_session.id,
        #         run_id=self.current_run_ids[user_id],
        #     )
        #     db_session.add(protocol_chosen)
        #     db_session.commit()
        #     self.current_protocol_ids[user_id] = [current_protocol, protocol_chosen.id]
        #
        #     for i in range(len(self.suggestions[user_id])):
        #         curr_protocols = self.suggestions[user_id][i]
        #         if curr_protocols[0] == self.PROTOCOL_TITLES[current_protocol]:
        #             curr_protocols.popleft()
        #             if len(curr_protocols) == 0:
        #                 self.suggestions[user_id].pop(i)
        #             break

        # PRE: User choice is string in ["Better", "Worse"]
        # elif current_choice == "user_found_useful":
        #     current_protocol = Protocol.query.filter_by(
        #         id=self.current_protocol_ids[user_id][1]
        #     ).first()
        #     current_protocol.protocol_was_useful = user_choice
        #     db_session.commit()

        if current_choice == "guess_emotion":
            option_chosen = user_choice + " ({})".format(
                self.guess_emotion_predictions[user_id]
            )
        else:
            option_chosen = user_choice
        choice_made = Choice(
            choice_desc=current_choice,
            option_chosen=option_chosen,
            user_id=user_id,
            session_id=user_session.id,
            run_id=self.current_run_ids[user_id],
        )
        db_session.add(choice_made)
        db_session.commit()

        return choice_made

    def determine_next_choice(
            self, user_id, input_type, user_choice, db_session, user_session, app
    ):
        # Find relevant user info by using user_id as key in dict.
        #
        # Then using the current choice and user input, we determine what the next
        # choice is and return this as the output.

        # Some edge cases to consider based on the different types of each field:
        # May need to return list of model responses. For next protocol, may need
        # to call function if callable.

        # If we cannot find the specific choice (or if None etc.) can set user_choice
        # to "any".

        # PRE: Will be defined by save_current_choice if it did not already exist.
        # (so cannot be None)

        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        current_choice_for_question = self.QUESTIONS[current_choice]["choices"]
        current_protocols = self.QUESTIONS[current_choice]["protocols"]
        if input_type != "open_text":
            # if (
            #         current_choice != "suggestions"
            #         and current_choice != "event_is_recent"
            #         and current_choice != "after_classification_positive"
            #         and current_choice != "user_found_useful"
            #         and current_choice != "check_emotion"
            #         and current_choice != "new_protocol_better"
            #         and current_choice != "new_protocol_worse"
            #         and current_choice != "new_protocol_same"
            #         # and current_choice != "choose_persona"
            #         and current_choice != "project_emotion"
            #         and current_choice != "after_classification_negative"
            #         and current_choice != "more_questions"
            # ):
            #     user_choice = user_choice.lower()

            # if (
            #         current_choice == "suggestions"
            # ):
            #     try:
            #         current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
            #     except KeyError:
            #         current_protocol = int(user_choice)
            #     protocol_choice = self.PROTOCOL_TITLES[current_protocol]
            #     next_choice = current_choice_for_question[protocol_choice]
            #     protocols_chosen = current_protocols[protocol_choice]

            # elif current_choice == "check_emotion":
            #     if user_choice == "Positive":
            #         next_choice = current_choice_for_question["Positive"]
            #         protocols_chosen = current_protocols["Positive"]
            #     else:
            #         next_choice = current_choice_for_question["Negative"]
            #         protocols_chosen = current_protocols["Negative"]
            if current_choice == "check_emotion":
                if user_choice == "Sad":
                    next_choice = current_choice_for_question["Sad"]
                    protocols_chosen = current_protocols["Sad"]
                elif user_choice == "Angry":
                    next_choice = current_choice_for_question["Angry"]
                    protocols_chosen = current_protocols["Angry"]
                elif user_choice == "Anxious/Scared":
                    next_choice = current_choice_for_question["Anxious/Scared"]
                    protocols_chosen = current_protocols["Anxious/Scared"]
                elif user_choice == "Others, but positive feelings":
                    next_choice = current_choice_for_question["Others, but positive feelings"]
                    protocols_chosen = current_protocols["Others, but positive feelings"]
                elif user_choice == "Others, but negative feelings":
                    next_choice = current_choice_for_question["Others, but negative feelings"]
                    protocols_chosen = current_protocols["Others, but negative feelings"]
                else:
                    next_choice = current_choice_for_question["Happy/Content"]
                    protocols_chosen = current_protocols["Happy/Content"]
            else:
                print(user_choice)
                next_choice = current_choice_for_question[user_choice]
                protocols_chosen = current_protocols[user_choice]

        else:
            next_choice = current_choice_for_question["open_text"]
            protocols_chosen = current_protocols["open_text"]

        if callable(next_choice):
            next_choice = next_choice(user_id, db_session, user_session, app)

        if current_choice == "guess_emotion" and user_choice == "Yes":
            #     if self.guess_emotion_predictions[user_id] == "positive":
            #         next_choice = next_choice["Positive"]
            #     else:
            #         next_choice = next_choice["Negative"]
            if self.guess_emotion_predictions[user_id] == "sad":
                next_choice = next_choice["Sad"]
            elif self.guess_emotion_predictions[user_id] == "angry":
                next_choice = next_choice["Angry"]
            elif self.guess_emotion_predictions[user_id] == "fear":
                next_choice = next_choice["Anxious/Scared"]
            else:
                next_choice = next_choice["Happy/Content"]

        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(user_id, db_session, user_session, app)
        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]
        if callable(next_prompt):
            next_prompt = next_prompt(user_id, db_session, user_session, app)
        # if (
        #         len(protocols_chosen) > 0
        #         and current_choice != "suggestions"
        # ):
        #     self.update_suggestions(user_id, protocols_chosen, app)

        # Case: new suggestions being created after first protocol attempted
        if next_choice == "opening_prompt":
            self.clear_suggestions(user_id)
            self.clear_emotion_scores(user_id)
            self.create_new_run(user_id, db_session, user_session)

        # if next_choice == "suggestions":
        #     next_choices = self.get_suggestions(user_id, app)

        # else:
        next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())
        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice
        return {"model_prompt": next_prompt, "choices": next_choices}
