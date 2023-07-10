# -*- coding: utf-8 -*-
"""module for LV2URLHISTORYAnalyzer."""
import os
from advanced_modules import manager
from advanced_modules import interface
from advanced_modules import logger

"""for crawler"""
import bs4
import googletrans  # pip3 install googletrans==4.0.0-rc1
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from keybert import KeyBERT
from cleantext import clean
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

"""for model"""
import joblib
import fasttext.util
import numpy as np


class Crawler:
    def __init__(self, url):
        self.keyword = None
        self.text = None
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = TreebankWordTokenizer()
        self.translator = googletrans.Translator()
        self.stop_words_list = stopwords.words('english')
        self.url = url
        options = webdriver.ChromeOptions()
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
        options.add_argument('user-agent=' + user_agent)
        options.add_argument("lang=en_US")
        options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument('--log-level=3')
        options.add_argument("--disable-logging")
        options.add_argument("--disable-logging-redirect")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        self.options = options

    def request(self):
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        driver.implicitly_wait(10)  # dynamic page wait

        try:
            driver.get(self.url)
            driver.execute_script(
                "Object.defineProperty(navigator, 'plugins', {get: function() {return[1, 2, 3, 4, 5]}})")
        except Exception as e:
            self.text = ""
            driver.quit()
            return
        html = driver.page_source
        soup = bs4.BeautifulSoup(html, 'html.parser')
        for s in soup.select('script'):
            s.extract()
        for s in soup.select('style'):
            s.extract()
        for s in soup.select('image'):
            s.extract()
        self.text = soup.get_text("\n")
        driver.quit()

    def translate(self):
        if self.text is None or len(self.text) == 0:
            return
        translated = ''
        # translate to english maximum 5000 characters in one request
        newline = ''
        for line in self.text.splitlines():
            line = line.strip()
            if len(line) == 0 or len(line) >= 4096:
                continue
            if len(newline) + len(line) >= 4000:
                try:
                    translated += self.translator.translate(newline, dest='en').text + '\n'
                except Exception as e:
                    print("[MODULE] Module for LV2 URL History Analzer: error: translate: ", e, "\n")
                    translated += newline + '\n'
                newline = line
            else:
                newline = newline + "\n" + line
        if len(newline) > 0:
            try:
                translated += self.translator.translate(newline, dest='en').text + '\n'
            except Exception as e:
                print("[MODULE] Module for LV2 URL History Analzer: error: translate: ", e, "\n")
                translated += newline + '\n'
        self.text = translated

    def preprocess(self):
        if self.text is None or len(self.text) == 0:
            self.keyword = []
            return
        self.text = clean(text=self.text,
                          fix_unicode=True,
                          to_ascii=True,
                          lower=True,
                          no_line_breaks=False,
                          no_urls=True,
                          no_emails=True,
                          no_phone_numbers=True,
                          no_numbers=True,
                          no_digits=True,
                          no_currency_symbols=True,
                          no_punct=True,
                          lang="en")
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(self.text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10)

        # extract keyword only
        self.keyword = [w[0] for w in keywords]

    def crawling_data_from_url(self):
        # crawler = Crawler(url)
        print(f"[MODULE] LV2 Url History Analyzer - crawling data from url: {self.url}")
        self.request()
        self.translate()
        self.preprocess()
        return self.keyword


class ModelPredictor:
    def __init__(self):
        self.fasttext_model = None
        self.keywords = None
        this_file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'models' + os.sep

        # model 파일
        model_file = this_file_path + 'lv2_url_history_model.joblib'
        self.model = joblib.load(model_file)

    def set_keywords(self, keywords):
        self.keywords = keywords

    def load_fasttext_model(self):
        print('[MODULE] LV2 Url History Analyzer - start load fasttext model')
        fasttext.util.download_model('en', if_exists='ignore')
        self.fasttext_model = fasttext.load_model('cc.en.300.bin')
        print('[MODULE] LV2 Url History Analyzer -   end load fasttext model')

    def get_feature(self):
        # get similar word vector
        if self.keywords is None or len(self.keywords) == 0:
            avg_vector = np.zeros(300)
        else:
            keywords_vector = []
            for i in range(len(self.keywords)):
                keywords_vector.append(self.fasttext_model.get_word_vector(self.keywords[i]))

            # get average vector of similar word vector in 300 dimension
            avg_vector = np.zeros(300)
            for i in range(len(keywords_vector)):
                avg_vector += keywords_vector[i]
            avg_vector /= len(keywords_vector)
            avg_vector = avg_vector.tolist()
        return avg_vector


class LV2URLHISTORYAnalyzer(interface.AdvancedModuleAnalyzer):
    NAME = 'lv2_url_history_analyzer'
    DESCRIPTION = 'Module for LV2 URL History Analyzer'

    _plugin_classes = {}

    def __init__(self):
        super(LV2URLHISTORYAnalyzer, self).__init__()
        self.model = None

    def get_category(self, url):
        categories = ["arts-and-entertainment", "business-and-consumer-services", "community-and-society",
                      "computers-electronics-and-technology", "e-commerce-and-shopping", "finance", "food-and-drink",
                      "gambling", "games", "health", "heavy-industry-and-engineering", "hobbies-and-leisure",
                      "home-and-garden", "jobs-and-career", "law-and-government", "lifestyle", "news-and-media",
                      "pets-and-animals", "reference-materials", "science-and-education", "sports",
                      "travel-and-tourism", "vehicles", "adult"]
        crawler = Crawler(url)
        keywords = crawler.crawling_data_from_url()

        self.model.set_keywords(keywords)
        keyword_str = ' '.join(keywords)
        if len(keywords) == 0 or 'cookie' in keyword_str or 'bot' in keyword_str or 'cloudflare' in keyword_str or 'forbidden' in keyword_str or 'captcha' in keyword_str:
            return 'unknown'
        feature = self.model.get_feature()
        if (feature == np.zeros(300)).all():
            return 'unknown'
        category = self.model.model.predict([feature])[0]
        return categories[category]

    def Analyze(self, par_id, configuration, source_path_spec, knowledge_base):
        print('[MODULE] LV2 Url History Analyzer')
        self.model = ModelPredictor()
        self.model.load_fasttext_model()

        this_file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'schema' + os.sep

        # 모든 yaml 파일 리스트
        yaml_list = [this_file_path + 'lv2_url_history.yaml']

        # 모든 테이블 리스트
        table_list = ['lv2_url_history']

        # 모든 테이블 생성
        for count in range(0, len(yaml_list)):
            if not self.LoadSchemaFromYaml(yaml_list[count]):
                logger.error('cannot load schema from yaml: {0:s}'.format(table_list[count]))
                return False

            # if table is not existed, create table
            if not configuration.cursor.check_table_exist(table_list[count]):
                ret = self.CreateTable(configuration.cursor)
                if not ret:
                    logger.error('cannot create database table name: {0:s}'.format(table_list[count]))
                    return False

        # source table(lv1_app_web_*_visit_urls)
        source_table_list = ['lv1_app_web_chrome_visit_urls',
                             'lv1_app_web_chromium_edge_visit_urls',
                             'lv1_app_web_firefox_visit_urls',
                             'lv1_app_web_opera_visit_urls',
                             'lv1_app_web_whale_visit_urls']

        # SELECT all rows from source tables
        insert_data = []
        for source_table in source_table_list:
            # SELECT all rows from source table
            print('[MODULE] LV2 Url History Analyzer - select all rows from source table: {0:s}'.format(source_table))
            query = f"SELECT par_id, case_id, evd_id, url, last_visited_time, title, visit_count, typed_count, os_account, source " \
                    f"FROM {source_table};"
            results = configuration.cursor.execute_query_mul(query)
            try:
                if len(results) == 0:
                    continue
                # INSERT rows into lv2_url_history table
                for result in results:
                    par_id = result[0]
                    case_id = result[1]
                    evd_id = result[2]
                    url = result[3]
                    last_visited_time = result[4]
                    title = result[5]
                    visit_count = result[6]
                    typed_count = result[7]
                    os_account = result[8]
                    source = result[9]
                    category = self.get_category(url)
                    insert_data.append(tuple(
                        [par_id, case_id, evd_id, url, last_visited_time, title, visit_count, typed_count, os_account,
                         source, category]))
            except:
                continue
        query = "INSERT INTO lv2_url_history values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        configuration.cursor.bulk_execute(query, insert_data)
        return True


manager.AdvancedModulesManager.RegisterModule(LV2URLHISTORYAnalyzer)
