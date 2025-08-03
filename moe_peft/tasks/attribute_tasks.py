import logging
import random
from typing import List, Optional

import datasets as hf_datasets
import torch
import json
import re
import os
from tqdm import tqdm

from transformers import BertTokenizer, BertModel


from moe_peft.common import InputData

from moe_peft.tasks.common import AttributeTask, BasicMetric, AutoMetric


class AttributedAnswerTask(AttributeTask):
    def __init__(self) -> None:
        super().__init__()      
        

    def loading_metric(self, metrics: List[str]):

        return AutoMetric("attribute", metrics)

class ASQA(AttributedAnswerTask):
    def __init__(self, sub: str = 'vani'):
        super().__init__()
        self.inst = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.inst_special_token = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.inst_new = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite all of them at the end of the sentences. Use an unbiased and journalistic tone. Always cite for any factual claim. Cite at least one document in each sentence.'
        self.sub = sub

    def loading_data(self, is_train: bool = False, path: str = None, few_shot: bool = True
                     ) -> List[InputData]:
        few_shot = False #################################

        num_docs = 5
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../../dataset/ALCE-data/asqa_eval_gtr_top100.json"  # 向上两级再进入dataset目录
        file_path = os.path.join(current_dir, relative_path)

        with open(path if path is not None else file_path,'r',encoding='utf-8') as file:
            data = json.load(file)
        logging.info("Preparing data for ASQA")
        ret: List[InputData] = []
        #cnt = 5
        """tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained('bert-large-uncased')
        device = 'cuda:6'
        model = model.to(device)
        model.eval()"""
        for data_point in tqdm(data):
            #if cnt == 0:
            #    break
            #cnt = cnt - 1 
            #prompt = ""
            prompt = "<|start_header_id|>system<|end_header_id|>\n\n" + "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            #prompt += self.inst_new
            prompt += self.inst_special_token
            if few_shot:
                prompt += f"Here is an example:\n\nQuestion: Who played galen in planet of the apes?\n\nDocument [1](Title: Planet of the Apes): installment. Jacobs died on June 27, 1973, bringing an end to the APJAC Productions era of the \"Planet of the Apes\" franchise. Former Fox executive Stan Hough took over as producer for the television project, titled \"Planet of the Apes\". CBS picked up the series for its 1974 autumn lineup. Ron Harper and James Naughton played Alan Virdon and Peter Burke, two 20th-century American astronauts who pass through a time warp to a future where apes subjugate humans (unlike the original film, the humans can speak). Roddy McDowall returned to the franchise as Galen, a chimpanzee who joins the astronauts.\nDocument [2](Title: Planet of the Apes (1968 film)): chimpanzees: animal psychologist Zira (Kim Hunter) and surgeon Galen (Wright King). While unable to speak as his throat wound is healing, called \"Bright Eyes\" by Zira and placed with one of the captive primitive humans he later names \"Nova\", Taylor observes the enhanced society of talking apes and in a strict caste system: the gorillas being the military police, hunters and workers; the orangutans overseeing the affairs of government, science, and religion; and intellectual chimpanzees being mostly scientists. While their society is a theocracy similar to the beginnings of the human Industrial Era, the apes consider the primitive humans as\nDocument [3](Title: Planet of the Apes (1968 film)): Planet of the Apes (1968 film) Planet of the Apes is a 1968 American science fiction film directed by Franklin J. Schaffner. It stars Charlton Heston, Roddy McDowall, Kim Hunter, Maurice Evans, James Whitmore, James Daly and Linda Harrison. The screenplay by Michael Wilson and Rod Serling was loosely based on the 1963 French novel \"La Plan\u00e8te des Singes\" by Pierre Boulle. Jerry Goldsmith composed the groundbreaking avant-garde score. It was the first in a series of five films made between 1968 and 1973, all produced by Arthur P. Jacobs and released by 20th Century Fox. The film tells the\nDocument [4](Title: Planet of the Apes): Rupert Wyatt. To portray ape characters realistically, the production avoided practical effects in favor of performance capture acting, partnering with New Zealand visual effects company Weta Digital. Wyatt cast James Franco as Will Rodman, while veteran performance capture actor Andy Serkis signed on to star as Caesar. \"Rise\" debuted on August 5, 2011. Critics reviewed it positively, especially praising the visual effects and Serkis's performance. It was a major box office hit, taking in $482 million globally, more than five times its $93 million budget. Weta's special effects earned the film two Visual Effects Society Awards and an Oscar nomination\nDocument [5](Title: Planet of the Apes): film stars Mark Wahlberg as astronaut Leo Davidson, who accidentally travels through a wormhole to a distant planet where talking apes enslave humans. He leads a human revolt and upends ape civilization by discovering that the apes evolved from the normal earth primates who had accompanied his mission, and arrived years before. Helena Bonham Carter played chimpanzee Ari, while Tim Roth played the human-hating chimpanzee General Thade. The film received mixed reviews; most critics believed it failed to compare to the original. Much of the negative commentary focused on the confusing plot and twist ending, though many reviewers praised the\n\nAnswer:In the 1968 film Planet of the Apes, Galen was played by Wright King [2]. And in the tv series Planet of the Apes, Galen was played by Roddy McDowall [1].\n\n\n"
            #prompt += f"\n\n\nQusetion: {data_point['qa_pairs'][0]['question']}\n\n"
            prompt += f"\n\n\nQusetion: {data_point['question']}\n\n"
            docs = ""
            cites = []
            for i in range(num_docs):
                cites.append({
                    'text': data_point['docs'][i]['text'],
                    'title': data_point['docs'][i]['title'],
                    'summary': data_point['docs'][i]['summary'],
                    })
            #random.shuffle(cites)
            for i in range(num_docs):
                docs += f"Document <|reserved_special_token_{i+1}|>: {cites[i]['text'] if self.sub=='vani' else cites[i]['summary']}\n"
                #docs += f"Document <|reserved_special_token_{i+1}|>(Title: {cites[i]['title']}): {cites[i]['text'] if self.sub=='vani' else cites[i]['summary']}\n"
                #docs += f"Document [{i+1}](Title: {cites[i]['title']}): {cites[i]['text'] if self.sub=='vani' else cites[i]['summary']}\n"
            cites = [cites[i]['text'] if self.sub=='vani' else cites[i]['summary'] for i in range(num_docs)]
            prompt += docs
            prompt += f"\nAnswer:"
            # prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            #citation_embeds = sents_embed(cites, model, tokenizer, device)
            ret.append(InputData(inputs=prompt, labels=data_point['answer'], \
                                 grounds=data_point['qa_pairs'], citations = cites,# citation_embeds = citation_embeds,\
                                query = data_point['question']))

        return ret
    
    def loading_metric(self):
        config = {}
        config['task'] = 'asqa'
        config['metric'] = metric_list['asqa']
        return AutoMetric("attribute", config)


class ELI5(AttributedAnswerTask):
    def __init__(self, sub: str = 'vani'):
        super().__init__()
        self.inst = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.inst_special_token = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.sub = sub

    def loading_data(self, is_train: bool = False, path: str = None, few_shot: bool = True
                     ) -> List[InputData]:
        few_shot = False ##############
        num_docs = 5
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../../dataset/ALCE-data/eli5_eval_bm25_top100.json"  # 向上两级再进入dataset目录
        file_path = os.path.join(current_dir, relative_path)
        with open(path if path is not None else file_path,'r',encoding='utf-8') as file:
            data = json.load(file)
        logging.info("Preparing data for ELI5")
        ret: List[InputData] = []
        #cnt = 5
        for data_point in tqdm(data):
            #if cnt == 0:
            #    break
            #cnt = cnt - 1 
            prompt = "<|start_header_id|>system<|end_header_id|>\n\n" + "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            #prompt += self.inst
            prompt += self.inst_special_token
            if few_shot:
                prompt += f"Here is an example:\n\nQuestion: Who played galen in planet of the apes?\n\nDocument [1](Title: Planet of the Apes): installment. Jacobs died on June 27, 1973, bringing an end to the APJAC Productions era of the \"Planet of the Apes\" franchise. Former Fox executive Stan Hough took over as producer for the television project, titled \"Planet of the Apes\". CBS picked up the series for its 1974 autumn lineup. Ron Harper and James Naughton played Alan Virdon and Peter Burke, two 20th-century American astronauts who pass through a time warp to a future where apes subjugate humans (unlike the original film, the humans can speak). Roddy McDowall returned to the franchise as Galen, a chimpanzee who joins the astronauts.\nDocument [2](Title: Planet of the Apes (1968 film)): chimpanzees: animal psychologist Zira (Kim Hunter) and surgeon Galen (Wright King). While unable to speak as his throat wound is healing, called \"Bright Eyes\" by Zira and placed with one of the captive primitive humans he later names \"Nova\", Taylor observes the enhanced society of talking apes and in a strict caste system: the gorillas being the military police, hunters and workers; the orangutans overseeing the affairs of government, science, and religion; and intellectual chimpanzees being mostly scientists. While their society is a theocracy similar to the beginnings of the human Industrial Era, the apes consider the primitive humans as\nDocument [3](Title: Planet of the Apes (1968 film)): Planet of the Apes (1968 film) Planet of the Apes is a 1968 American science fiction film directed by Franklin J. Schaffner. It stars Charlton Heston, Roddy McDowall, Kim Hunter, Maurice Evans, James Whitmore, James Daly and Linda Harrison. The screenplay by Michael Wilson and Rod Serling was loosely based on the 1963 French novel \"La Plan\u00e8te des Singes\" by Pierre Boulle. Jerry Goldsmith composed the groundbreaking avant-garde score. It was the first in a series of five films made between 1968 and 1973, all produced by Arthur P. Jacobs and released by 20th Century Fox. The film tells the\nDocument [4](Title: Planet of the Apes): Rupert Wyatt. To portray ape characters realistically, the production avoided practical effects in favor of performance capture acting, partnering with New Zealand visual effects company Weta Digital. Wyatt cast James Franco as Will Rodman, while veteran performance capture actor Andy Serkis signed on to star as Caesar. \"Rise\" debuted on August 5, 2011. Critics reviewed it positively, especially praising the visual effects and Serkis's performance. It was a major box office hit, taking in $482 million globally, more than five times its $93 million budget. Weta's special effects earned the film two Visual Effects Society Awards and an Oscar nomination\nDocument [5](Title: Planet of the Apes): film stars Mark Wahlberg as astronaut Leo Davidson, who accidentally travels through a wormhole to a distant planet where talking apes enslave humans. He leads a human revolt and upends ape civilization by discovering that the apes evolved from the normal earth primates who had accompanied his mission, and arrived years before. Helena Bonham Carter played chimpanzee Ari, while Tim Roth played the human-hating chimpanzee General Thade. The film received mixed reviews; most critics believed it failed to compare to the original. Much of the negative commentary focused on the confusing plot and twist ending, though many reviewers praised the\n\nAnswer:In the 1968 film Planet of the Apes, Galen was played by Wright King [2]. And in the tv series Planet of the Apes, Galen was played by Roddy McDowall [1].\n\n\n"
            prompt += f"\n\n\nQusetion: {data_point['question']}\n\n"
            docs = ""
            cites = []
            for i in range(num_docs):
                cites.append({
                    'text': data_point['docs'][i]['text'],
                    'title': data_point['docs'][i]['title'],
                    'summary': data_point['docs'][i]['summary'],
                    })
            #random.shuffle(cites)
            for i in range(num_docs):
                docs += f"Document <|reserved_special_token_{i+1}|>: {cites[i]['text'] if self.sub=='vani' else cites[i]['summary']}\n"
                #docs += f"Document [{i+1}](Title: {cites[i]['title']}): {cites[i]['text'] if self.sub=='vani' else cites[i]['summary']}\n"
            cites = [cites[i]['text'] if self.sub=='vani' else cites[i]['summary'] for i in range(num_docs)]
            prompt += docs
            prompt += f"\nAnswer:"
            # prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            ret.append(InputData(inputs=prompt, labels=data_point['answer'], \
                                 grounds=data_point['claims'], citations = cites, \
                                query = data_point['question']))

        return ret
    
    def loading_metric(self):
        config = {}
        config['task'] = 'eli5'
        config['metric'] = metric_list['eli5']
        return AutoMetric("attribute", config)
    
class Qampari(AttributedAnswerTask):
    def __init__(self, sub: str = 'vani'):
        super().__init__()
        self.inst = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.inst_special_token = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.sub = sub

    def loading_data(self, is_train: bool = False, path: str = None, few_shot: bool = True
                     ) -> List[InputData]:
        few_shot = False ##############
        num_docs = 5
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../../dataset/ALCE-data/qampari_eval_gtr_top100.json"  # 向上两级再进入dataset目录
        file_path = os.path.join(current_dir, relative_path)
        with open(path if path is not None else file_path,'r',encoding='utf-8') as file:
            data = json.load(file)
        logging.info("Preparing data for Qampari")
        ret: List[InputData] = []
        #cnt = 5
        for data_point in tqdm(data):
            #if cnt == 0:
            #    break
            #cnt = cnt - 1 
            prompt = "<|start_header_id|>system<|end_header_id|>\n\n" + "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            #prompt += self.inst
            prompt += self.inst_special_token
            if few_shot:
                prompt += f"Here is an example:\n\nQuestion: Who played galen in planet of the apes?\n\nDocument [1](Title: Planet of the Apes): installment. Jacobs died on June 27, 1973, bringing an end to the APJAC Productions era of the \"Planet of the Apes\" franchise. Former Fox executive Stan Hough took over as producer for the television project, titled \"Planet of the Apes\". CBS picked up the series for its 1974 autumn lineup. Ron Harper and James Naughton played Alan Virdon and Peter Burke, two 20th-century American astronauts who pass through a time warp to a future where apes subjugate humans (unlike the original film, the humans can speak). Roddy McDowall returned to the franchise as Galen, a chimpanzee who joins the astronauts.\nDocument [2](Title: Planet of the Apes (1968 film)): chimpanzees: animal psychologist Zira (Kim Hunter) and surgeon Galen (Wright King). While unable to speak as his throat wound is healing, called \"Bright Eyes\" by Zira and placed with one of the captive primitive humans he later names \"Nova\", Taylor observes the enhanced society of talking apes and in a strict caste system: the gorillas being the military police, hunters and workers; the orangutans overseeing the affairs of government, science, and religion; and intellectual chimpanzees being mostly scientists. While their society is a theocracy similar to the beginnings of the human Industrial Era, the apes consider the primitive humans as\nDocument [3](Title: Planet of the Apes (1968 film)): Planet of the Apes (1968 film) Planet of the Apes is a 1968 American science fiction film directed by Franklin J. Schaffner. It stars Charlton Heston, Roddy McDowall, Kim Hunter, Maurice Evans, James Whitmore, James Daly and Linda Harrison. The screenplay by Michael Wilson and Rod Serling was loosely based on the 1963 French novel \"La Plan\u00e8te des Singes\" by Pierre Boulle. Jerry Goldsmith composed the groundbreaking avant-garde score. It was the first in a series of five films made between 1968 and 1973, all produced by Arthur P. Jacobs and released by 20th Century Fox. The film tells the\nDocument [4](Title: Planet of the Apes): Rupert Wyatt. To portray ape characters realistically, the production avoided practical effects in favor of performance capture acting, partnering with New Zealand visual effects company Weta Digital. Wyatt cast James Franco as Will Rodman, while veteran performance capture actor Andy Serkis signed on to star as Caesar. \"Rise\" debuted on August 5, 2011. Critics reviewed it positively, especially praising the visual effects and Serkis's performance. It was a major box office hit, taking in $482 million globally, more than five times its $93 million budget. Weta's special effects earned the film two Visual Effects Society Awards and an Oscar nomination\nDocument [5](Title: Planet of the Apes): film stars Mark Wahlberg as astronaut Leo Davidson, who accidentally travels through a wormhole to a distant planet where talking apes enslave humans. He leads a human revolt and upends ape civilization by discovering that the apes evolved from the normal earth primates who had accompanied his mission, and arrived years before. Helena Bonham Carter played chimpanzee Ari, while Tim Roth played the human-hating chimpanzee General Thade. The film received mixed reviews; most critics believed it failed to compare to the original. Much of the negative commentary focused on the confusing plot and twist ending, though many reviewers praised the\n\nAnswer:In the 1968 film Planet of the Apes, Galen was played by Wright King [2]. And in the tv series Planet of the Apes, Galen was played by Roddy McDowall [1].\n\n\n"
            prompt += f"\n\n\nQusetion: {data_point['question']}\n\n"
            docs = ""
            cites = []
            for i in range(num_docs):
                cites.append({
                    'text': data_point['docs'][i]['text'],
                    'title': data_point['docs'][i]['title'],
                    })
            #random.shuffle(cites)
            for i in range(num_docs):
                docs += f"Document <|reserved_special_token_{i+1}|>: {cites[i]['text']}\n"
                #docs += f"Document [{i+1}](Title: {cites[i]['title']}): {cites[i]['text'] if self.sub=='vani' else cites[i]['summary']}\n"
            cites = [cites[i]['text'] for i in range(num_docs)]
            prompt += docs
            prompt += f"\nAnswer:"
            # prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            ret.append(InputData(inputs=prompt, labels=data_point['answers'], \
                                 citations = cites, \
                                query = data_point['question']))
        return ret
    
    def loading_metric(self):
        config = {}
        config['task'] = 'qam'
        config['metric'] = metric_list['qam']
        return AutoMetric("attribute", config)


class QouteSum(AttributedAnswerTask):
    def __init__(self, sub: str = 'vani'):
        super().__init__()
        self.sub = sub
        self.inst = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.inst2 = 'Based on the information contained in the document, answer the question with details to the best of your bilities. Think step by step and explain your answer if that will help better understand the answer.'
        self.inst_special_token = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.inst_new = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite all of them at the end of the sentences. Use an unbiased and journalistic tone. Always cite for any factual claim. Cite at least one document in each sentence.'

    def loading_data(self, is_train: bool = False, path: str = None,
                    few_shot: bool = True ) -> List[InputData]:
        few_shot = False ###########
        if is_train:
            few_shot = False
        ret: List[InputData] = []
        examples_by_qid = {}
        """tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained('bert-large-uncased')
        device = 'cuda:6'
        model = model.to(device)
        model.eval()"""
        with open(f"/yy21/MoE-PEFT/dataset/{'qoutesum_alce' if self.sub == 'alce' else ( 'qoutesum_ans' if self.sub == 'ans' else 'qoutesum')}/{'train' if is_train else 'test'}.jsonl" if path is None else path, 'r') as f:
            #cnt = 50
            for line in f:
                #if cnt == 0:
                #    break
                #cnt -= 1
                example = json.loads(line.strip())
                if example['qid'] not in examples_by_qid:
                    examples_by_qid[example['qid']] = [example]
                else:
                    examples_by_qid[example['qid']].append(example)

        examples = list(examples_by_qid.values()) 
        for example in examples:
            prompt = "<|start_header_id|>system<|end_header_id|>\n\n" + "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            prompt += self.inst_special_token
            #prompt += self.inst_new
            if few_shot:
                if self.sub == 'alce':
                    prompt += f" Here are some examples:\nQuestion: how much power does a wind turbine produce?\nDocument [1](Title:): Compact wind acceleration turbine: It is generally thought that since the amount of power produced by a wind turbine is proportional to the cube of the wind speed, any acceleration benefit is potentially statistically significant in the economics of wind. As noted though this is an inaccurate as it ignores the impact of the exit to area ratio and is therefore an apples to oranges comparison. In the case of a typical CWAT/DAWT the power result in perfect theoretical operation once adjusted for the area of the shroud is actually the square of the velocity at the rotor. As the CWAT/DAWT diverges from theoretical function the power increase drops significantly according\nDocument [2](Title:): Sustainable architecture: roof ledge. Small-scale rooftop wind turbines have been known to be able to generate power from 10% to up to 25% of the electricity required of a regular domestic household dwelling. Turbines for residential scale use are usually between 7 feet (2 m) to 25 feet (8 m) in diameter and produce electricity at a rate of 900 watts to 10,000 watts at their tested wind speed. Building integrated wind turbine performance can be enhanced with the addition of an aerofoil wing on top of a roof mounted turbine. Solar water heaters, also called solar domestic hot water systems, can\nDocument [3](Title:): Turby wind turbine: can because horizontal axis (HAWT) types cannot change their pitch to face the wind directly. The turbine measures 2.0m (6'7\") in diameter by 2.9m (9'6\") high (including generator), and weighs 136 kg (300 lb). It is specified to generate power in winds of between 4 m/s (9 mph, 7.8kts) and 14 m/s (31 mph, 27.2kts), and can survive winds of 55 m/s (123 mph, 107kts). The rated power at 14 m/s is 2.5 kW (3.35 hp). The AC output from the synchronous generator is rectified to DC, then inverted to AC at 230V 50 Hz. Core International developed the turbine\nAnswer: One source states the amount of power produced by a wind turbine is proportional to the cube of the wind speed [1]. Other sources state Turbines for residential scale use produce electricity at a rate of 900 watts to 10,000 watts, and is specified to generate power in winds of between 4 m/s (9 mph, 7.8kts) and 14 m/s (31 mph, 27.2kts) [2][3]."
                elif self.sub == 'vani':
                    prompt += f" Here are some examples:\nQuestion: how much power does a wind turbine produce?\n[1] Compact wind acceleration turbine: It is generally thought that since the amount of power produced by a wind turbine is proportional to the cube of the wind speed, any acceleration benefit is potentially statistically significant in the economics of wind. As noted though this is an inaccurate as it ignores the impact of the exit to area ratio and is therefore an apples to oranges comparison. In the case of a typical CWAT/DAWT the power result in perfect theoretical operation once adjusted for the area of the shroud is actually the square of the velocity at the rotor. As the CWAT/DAWT diverges from theoretical function the power increase drops significantly according\n[2] Sustainable architecture: roof ledge. Small-scale rooftop wind turbines have been known to be able to generate power from 10% to up to 25% of the electricity required of a regular domestic household dwelling. Turbines for residential scale use are usually between 7 feet (2 m) to 25 feet (8 m) in diameter and produce electricity at a rate of 900 watts to 10,000 watts at their tested wind speed. Building integrated wind turbine performance can be enhanced with the addition of an aerofoil wing on top of a roof mounted turbine. Solar water heaters, also called solar domestic hot water systems, can\n[3] Turby wind turbine: can because horizontal axis (HAWT) types cannot change their pitch to face the wind directly. The turbine measures 2.0m (6'7\") in diameter by 2.9m (9'6\") high (including generator), and weighs 136 kg (300 lb). It is specified to generate power in winds of between 4 m/s (9 mph, 7.8kts) and 14 m/s (31 mph, 27.2kts), and can survive winds of 55 m/s (123 mph, 107kts). The rated power at 14 m/s is 2.5 kW (3.35 hp). The AC output from the synchronous generator is rectified to DC, then inverted to AC at 230V 50 Hz. Core International developed the turbine\nAnswer: One source states the [ 1 amount of power produced by a wind turbine is proportional to the cube of the wind speed ] . Other sources state [ 2 Turbines for residential scale use ] [ 2 produce electricity at a rate of 900 watts to 10,000 watts ] , and [ 3 is specified to generate power in winds of between 4 m/s (9 mph, 7.8kts) and 14 m/s (31 mph, 27.2kts) ] .\n\nQuestion: a component is what?\n[1] Modular programming: in Dart, Go or Java) is sometimes used instead of module. In other implementations, this is a distinct concept; in Python a package is a collection of modules, while in Java 9 the introduction of the new module concept (a collection of packages with enhanced access control) is planned. Furthermore, the term \"package\" has other uses in software (for example .NET NuGet packages). A component is a similar concept, but typically refers to a higher level; a component is a piece of a whole system, while a module is a piece of an individual program. The scale of the term\n[2] Physical body: the system at a point in time changes from identifying the object to not identifying it. Also an object's identity is created at the first point in time that the simplest model of the system consistent with perception identifies it. An object may be composed of components. A component is an object completely within the boundary of a containing object. In classical mechanics a physical body is collection of matter having properties including mass, velocity, momentum and energy. The matter exists in a volume of three-dimensional space. This space is its extension. Under Newtonian gravity the gravitational field further away\nQuoted summary: [ 1 A component is a similar concept, but typically refers to a higher level; a component is a piece of a whole system, while a module is a piece of an individual program ] in terms of [ 1 Modular programming ] . Whereas in the [ 2 Physical body ] , a [ 2 component is an object completely within the boundary of a containing object ] ."
                elif self.sub == 'ans':
                    pass
            prompt += f"\n\nQusetion: {example[0]['question']}\n"
            docs = ""
            sources = []
            citations = []
            #fk = 0
            for i in range(8):
                if f"title{i+1}" not in example[0]:
                    break
                #if example[0][f'title{i+1}'] == "":
                #    fk = i
                sources.append({'title': example[0][f'title{i+1}'],
                    'doc': example[0][f"source{i+1}"]}
                    )
            #random.shuffle(sources[:fk])
            for i in range(8):
                if sources[i]['doc'] != "":
                    #docs += f"Document [{i+1}](Title: {sources[i]['title']}): {sources[i]['doc']}\n"
                    #docs += f"Document <|reserved_special_token_{i+1}|>(Title: {sources[i]['title']}): {sources[i]['doc']}\n"
                    docs += f"Document <|reserved_special_token_{i+1}|>: {sources[i]['doc']}\n"
                    citations.append(sources[i]['doc'])
                else:
                    break
            if len(citations) == 0:
                continue
            #citations = sents_embed(citations, model, tokenizer, device)
            prompt += docs
            prompt += f"\nAnswer:"
            if is_train:
                for e in example:
                    #ret.append(InputData(inputs = prompt + e['summary']))
                    ret.append(InputData(inputs = prompt + cite2token(e['summary']),
                                         citations=citations, prompt = prompt))
            else:
                ret.append(InputData(inputs=prompt, labels=[e['summary'] for e in example], \
                                    grounds=[i for e in example for i in e['covered_short_answers']], \
                                    citations=citations, query = example[0]['question']))
        return ret
    
    def loading_metric(self):
        config = {}
        config['task'] = 'qsum'
        if self.sub == 'alce':
            config['metric'] = metric_list['qsum-a']
        else:
            config['metric'] = metric_list['qsum']
        return AutoMetric("attribute", config)


class Front(AttributedAnswerTask):
    def __init__(self, sub):
        super().__init__()
        self.inst = 'Extract the relevant content from the provided documents and then use the extracted content to guide answer generation and cite the sources properly.'
        self.sub = sub

    def loading_data(self, is_train: bool = False, few_shot: bool = True
                     ) -> List[InputData]:
        few_shot = False ##############
        with open("/yy21/MoE-PEFT/dataset/front/sft.json" if self.sub == 'sft' else "/yy21/MoE-PEFT/dataset/front/dpo.json",'r',encoding='utf-8') as file:
            data = json.load(file)
        logging.info("Preparing data for Front")
        ret: List[InputData] = []
        #cnt = 2

        for data_point in data:
            if data_point['instruction'] != self.inst:
                continue
            #if cnt == 0:
            #    break
            #cnt = cnt - 1 
            prompt = "<|start_header_id|>system<|end_header_id|>\n\n" + "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            prompt += self.inst
            prompt += data_point['input']
            prompt += "\nAnswer:"
            prompt = cite2token(prompt)
            q_start = len("Question: ")
            q_end = data_point['input'].find("\n\n", q_start)
            q = data_point['input'][q_start:q_end]
            cites = []
            pattern = r"Document \[(\d+)\]: (.*?)(?=Document \[\d+\]:|$)"
            matches = re.findall(pattern, data_point['input'][q_end + 2:], re.DOTALL)
            cites = [content.strip() for _, content in matches]
            #random.shuffle(cites)
            ans_idx = data_point['output'].find("[ANSWER]")
            ans = cite2token(data_point['output'][ans_idx + len("[ANSWER]"):])
            if is_train:
                ret.append(InputData(inputs = prompt + ans, prompt = prompt, citations=cites))
            else:
                ret.append(InputData(inputs=prompt, labels=ans, \
                                    citations = cites, query = q))
        return ret
    
    def loading_metric(self):
        config = {}
        config['task'] = 'front'
        config['metric'] = metric_list['front']
        return AutoMetric("attribute", config)
    

class Synsciqa(AttributedAnswerTask):
    def __init__(self, sub):
        super().__init__()
        self.sub = sub
        self.inst = lambda query: f"Can you respond to the question {query} by only relying on the sources. Ignore all sources that do not provide an answer to the question.                    Do not include any knowledge from outside of these sources. Only write a single paragraph. Each sentence must end with the reference in the form of (author, year, page number). Stricly follow this format. Citing multiple sources in one sentence is not allowed.                    However, if no source addresses the question, admit truthfully that no answer can be given.                    Answer the question concisly and avoid being verbose."
        self.inst_a = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.inst_special_token = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.inst_new = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite all of them at the end of the sentences. Use an unbiased and journalistic tone. Always cite for any factual claim. Cite at least one document in each sentence.'


    def loading_data(self, is_train: bool = False, few_shot: bool = True
                     ) -> List[InputData]:
        few_shot = False ##############
        current_dir = os.path.dirname(os.path.abspath(__file__))
          # 向上两级再进入dataset目录
        
        if self.sub == 'synsci':
            relative_path = "../../dataset/SynSciQA/SynSciQA.json"
        elif self.sub == 'synsci+':
            relative_path = "../../dataset/SynSciQA/SynSciQA+.json"
        elif self.sub == 'synsci++':
            relative_path = "../../dataset/SynSciQA/SynSciQA++.json"
        file_path = os.path.join(current_dir, relative_path)
        with open(file_path, 'r',encoding='utf-8') as file:
            data = json.load(file)

        logging.info("Preparing data for SynsciQA")
        ret: List[InputData] = []
        #cnt = 10

        """tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained('bert-large-uncased')
        device = 'cuda:4'
        model = model.to(device)
        model.eval()"""
        for line in tqdm(data):
            #if cnt == 0:
            #    break
            #cnt -= 1
            data_point = line["instruction"]
            answer = line["response"]
            doc_start = data_point.find("[BEGIN OF SOURCES]")
            doc_end = data_point.find("[END OF SOURCES]")
            documents = data_point[doc_start + len("[BEGIN OF SOURCES]"): doc_end].strip().split("\n")
            assert len(documents) > 0, print(f"No docs detected!")
            
            data_point = data_point[doc_end + len("[END OF SOURCES]"):]
            pattern = r'"([^"]*)"'
            query = re.findall(pattern, data_point)
            #prompt = ""
            prompt = "<|start_header_id|>system<|end_header_id|>\n\n" + "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            #prompt += self.inst_special_token
            #prompt += self.inst_new
            #prompt += self.inst_a
            prompt += f"\n\nQuestion: {query[0]}\n"
            
            docs = ""
            citations = []
            index_map = []
            index_map2 = []
            for i, d in enumerate(documents):
                Ids = d[:d.find(":")]
                cont = d[d.find(":") + 2:]
                docs += f"Document <|reserved_special_token_{i+1}|>: {cont}\n"
                #docs += f"Document [{i+1}]: {cont}\n"
                citations.append(cont)
                index_map.append({'index': f"({Ids})", 'ID': f'<|reserved_special_token_{i+1}|>'})
                index_map2.append({'index': f"{Ids}", 'ID': f'<|reserved_special_token_{i+1}|>'})
                #index_map.append({'index': f"({Ids})", 'ID': f'[{i+1}]'})
            index_map = {item['index']: item['ID'] for item in index_map}
            index_map2 = {item['index']: item['ID'] for item in index_map2}
            prompt +=docs
            prompt += "\nAnswer:"
            pattern = re.compile('|'.join(map(re.escape, index_map)))
            answer = pattern.sub(lambda m: index_map[m.group()], answer)
            pattern = re.compile('|'.join(map(re.escape, index_map2)))
            answer = pattern.sub(lambda m: index_map2[m.group()], answer)
            pattern = r'\(\s*(<\|[^|]+\|>)\s*;\s*(<\|[^|]+\|>)\s*\)'
            answer = re.sub(pattern, r'\1\2', answer)

            pattern = r'<\|reserved_special_token_\d+\|>'
            if bool(re.search(pattern, answer)) == False:
                continue
            pattern = r"\((?:[^)]*,){2}[^)]*p\.[^)]*\)"
            fk = re.findall(pattern, answer)
            if fk:
                continue
            #print(f"inputs:{prompt}\nans:{answer}\ncite{citations}")
            #input()
            #citation_embeds = sents_embed(citations, model, tokenizer, device)
            if is_train:
                ret.append(InputData(
                    inputs = prompt + answer, citations = citations, prompt = prompt#, citation_embeds = citation_embeds,
                ))
        return ret
    
    def loading_metric(self):
        config = {}
        config['task'] = 'front'
        config['metric'] = metric_list['front']
        return AutoMetric("attribute", config)
    

class Reinf(AttributedAnswerTask):
    def __init__(self):
        super().__init__()
        self.inst_a = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'
        self.inst_special_token = 'Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'


    def loading_data(self, is_train: bool = False, few_shot: bool = True
                     ) -> List[InputData]:
        few_shot = False ##############
        with open("/yy21/MoE-PEFT/dataset/reinforcement/combined_train.json", 'r',encoding='utf-8') as file:
            data = json.load(file)
        logging.info("Preparing data for Reinforcement")
        ret: List[InputData] = []
        #cnt = 305

        for line in tqdm(data):
            #if cnt == 0:
            #    break
            
            answer = line["output"][0]
            if bool(re.search(r'\[(\d+)\]', answer)) == False:
                continue
            cs = re.findall(r'\[(\d+)\]', answer)
            if max(map(int, cs)) > len(line["docs"]):
                continue

            query = line["question"]

            documents = line["docs"]
            answer = self.get_ans(answer)

            prompt = "<|start_header_id|>system<|end_header_id|>\n\n" + "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            prompt += self.inst_special_token
            #prompt += self.inst_new
            #prompt += self.inst_a
            prompt += f"\n\nQuestion: {query}\n"
            
            docs = ""
            citations = []
            for i, d in enumerate(documents):
                docs += f"Document <|reserved_special_token_{i+1}|>: {d['text']}\n"
                citations.append(d["text"])
            prompt +=docs
            prompt += "\nAnswer:"
            #cnt -= 1
            if is_train:
                ret.append(InputData(
                    inputs = prompt + answer, citations = citations, prompt = prompt
                ))
        return ret
    
    def get_ans(self, sent):
        def replace_cite(x):
            i = x.group(1)
            return f"<|reserved_special_token_{i}|>"
        return re.sub(r'\[(\d+)\]', replace_cite, sent)

    def loading_metric(self):
        config = {}
        config['task'] = 'front'
        config['metric'] = metric_list['front']
        return AutoMetric("attribute", config)

def sents_embed(sents, model, tokenizer, device):
    embeds = []
    with torch.no_grad():
        for sent in sents:
            inputs = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
            inputs = inputs.to(device)
            output = model(**inputs)
            embeds.append(output.pooler_output)
    result = torch.stack(embeds).squeeze(1)
    return result

def cite2token(sent):
    pattern = r'\[(\d+)\]'
    ans = re.sub(pattern, r'<|reserved_special_token_\g<1>|>', sent)
    return ans

metric_list = {
    'asqa': ['cite_pr', 'length', 'short_ans'],
    'qsum': ['rouge_all', 'semqa_f1', 'semqa_short'],
    'qsum-a': ['rouge_all','semqa_short', 'cite_pr', 'length', 'semqa_f1'],
    'eli5': ['cite_pr', 'eli5_acc', 'length'],
    'qam': ['cite_pr', 'qampari'],
    'front': [],
}

def update_task_dict(task_dict):
    task_dict.update(
        {
            "asqa": ASQA(),
            "qsum": QouteSum('vani'),
            "qsum-a": QouteSum('alce'),
            "qsum-ans": QouteSum('ans'),
            "eli5": ELI5(),
            "front-s": Front('sft'),
            "front-d": Front('dpo'),
            "synsci": Synsciqa('synsci'),
            "synsci+": Synsciqa('synsci+'),
            "synsci++": Synsciqa('synsci++'),
            "rein": Reinf(),
            "qam": Qampari()
        }
    )

if __name__ == '__main__':
    asqa = QouteSum()
    asqa.loading_data()