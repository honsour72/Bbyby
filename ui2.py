import mailbox
import re
from tkinter import *
from tkinter import filedialog as fd

import pandas as pd
from html2text import html2text
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM


class MainRoot(Tk):
    def __init__(self):
        super().__init__()
        self.filename = None
        self.title("Юзер симпл интерфейс")
        self.geometry("400x100+100+100")
        self.spam = 0
        self.resizable(False, False)

        # open-file frame
        self.open_frame = LabelFrame(self, text="Чтение файлов")
        # self.open_frame.pack(side=TOP, padx=10)
        self.open_frame.pack(fill='x', padx=10)

        self.filename_label = Label(self.open_frame, text="Выберите файл . . .", relief=GROOVE, height=2)
        self.filename_label.pack(side=LEFT, padx=10, pady=10)
        self.open_button = Button(self.open_frame, text="Открыть", relief=RAISED, bd=4, command=self.open_mbox_file)
        self.open_button.pack(side=LEFT, padx=5)
        self.start_button = Button(self.open_frame, text="Начать", relief=RAISED, bd=4, command=self.run_process)
        self.start_button.pack(side=RIGHT, padx=10)

        self.mainloop()

    def set_statistic(self):
        # preparing results
        self.result_text_frame = LabelFrame(self, text="Статистика по результату")
        # self.result_text_label = Label(self.result_text_frame, text="Результат будет здесь", background='white', width=40, bd=4)
        self.result_text_label = Label(self.result_text_frame)
        self.result_text_label.pack()
        # self.result_text_frame.pack(padx=10)
        self.result_text_frame.pack(fill='x', padx=10)

        # self.finish_label = Label(self, text="Result will be here . . . ", width=40, bg="#999", justify=LEFT)
        # self.finish_label.pack()

    def set_canvas(self):
        # reslut label-frame
        self.canvas = Canvas(self, height=540)
        self.result_frame = LabelFrame(self.canvas, text="Таблица")
        # # self.result_frame.pack(fill='x')
        # # self.result_frame.pack(side=RIGHT, fill='y', padx=10)
        lbl_row = Label(self.result_frame)
        Label(lbl_row, text="X-UID письма").pack(side=LEFT, padx=20)
        Label(lbl_row, text="Статус письма").pack(side=RIGHT, padx=20)
        lbl_row.pack(fill='x', padx=15)
        # # self.finish_label = Label(self.result_frame, text="Result will be here . . . ", width=50, bg="#999", justify=LEFT)
        # # self.finish_label.pack(side=LEFT, fill='y')
        #
        self.myscrollbar = Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.myscrollbar.set)
        #
        self.myscrollbar.pack(side=RIGHT, fill="y")
        self.canvas.pack(fill='both', padx=10)
        # # self.canvas.pack(side=RIGHT, expand=True)
        #
        self.canvas.create_window((0, 0), window=self.result_frame, anchor='nw')
        # self.canvas.create_window((0, 0), window=self.result_frame)
        self.result_frame.bind("<Configure>", lambda event, canvas=self.canvas: self.onFrameConfigure(canvas))
        #
        # # myscrollbar=Scrollbar(self.result_frame, orient="vertical")
        # # myscrollbar.pack(side=RIGHT, fill='y')
        # # self.configure(yscrollcommand=myscrollbar.set)
        #
        # # preparing results
        # self.result_text_frame = LabelFrame(self, text="Статистика по результату")
        # # self.result_text_label = Label(self.result_text_frame, text="Результат будет здесь", background='white', width=40, bd=4)
        # self.result_text_label = Label(self.result_text_frame)
        # self.result_text_label.pack(padx=10, pady=5)
        # # self.result_text_frame.pack(padx=10)
        # self.result_text_frame.pack(padx=10, fill='x')
        #
        # # self.finish_label = Label(self, text="Result will be here . . . ", width=40, bg="#999", justify=LEFT)
        # # self.finish_label.pack()

    def onFrameConfigure(self, canvas):
        '''Reset the scroll region to encompass the inner frame'''
        canvas.configure(scrollregion=canvas.bbox("all"))
        # self.result_frame.configure(padx=10)
        # self.result_frame.config(width=350)

    def open_mbox_file(self):
        filetypes = (
            ('ALL FILES', "*.*"),
            ('text files', '*.mbox'),
        )

        self.filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)

        local_path = self.filename.split("/")[-1]

        self.filename_label.config(text=local_path)

    def run_process(self):
        self.start_button['state'] = 'disabled'
        self.geometry("550x715+100+100")
        self.set_statistic()
        self.set_canvas()
        # self.result_frame.pack_configure(fill='both', expand=True)

        test_df, train_df = self.convert_from_mbox_to_df(self.filename)
        df = self.set_training(test_df, train_df)
        # df = self.set_training("", "")
        self.update_finish_label(df)
        self.show_statistics(df)

    def convert_from_mbox_to_df(self,  testfile='../mboxes/test.mbox', trainfile='train.mbox'):
        def read_mbox(mb_file):
            mb = mailbox.mbox(mb_file)
            mbox_dict = {}

            for i, mail in enumerate(mb):
                mbox_dict[i] = {}
                for header in mail.keys():
                    mbox_dict[i][header] = mail[header]
                mbox_dict[i]['Text'] = mail.get_payload()
            df = pd.DataFrame.from_dict(mbox_dict, orient='index')
            return df

        def read_msg(mb_msg):
            if type(mb_msg) == list:
                mb = mailbox.mboxMessage(mb_msg[0])
                return mb.get_payload()
            else:
                return mb_msg

        # оформление датафреймов
        train = read_mbox(trainfile)
        test = read_mbox(testfile)
        test['Text'] = test['Text'].apply(read_msg)

        def subject_cleaner(text):
            text = text.str.strip()
            text = text.str.lower()
            return text

        def body_cleaner(text):
            text = text.str.replace(r'<[^>]+>', '')
            text = text.str.replace(r'{[^}]+}', '')
            text = text.str.replace(r'#message', '')
            text = text.str.replace(r'\n{1,}', '')
            text = text.str.replace(r'={1,}', ' ')
            text = text.str.replace(r'-{2,}', ' ')
            text = text.str.replace(r'\*{1,}', ' ')
            text = text.str.replace(r'&nbsp{1,}', ' ')
            text = text.str.replace(r'\t', ' ')
            text = text.str.replace(r'\s{1,}', ' ')
            text = text.str.strip()
            text = text.str.lower()
            return text

        train['Subject'] = subject_cleaner(train['Subject'])
        test['Subject'] = subject_cleaner(test['Subject'])
        train['Text'] = body_cleaner(train['Text'])
        test['Text'] = body_cleaner(test['Text'])

        train['Content-Type'] = train['Content-Type'].apply(lambda x: x.lower().split(';')[0])

        test['Content-Type'] = test['Content-Type'].str.lower().str.strip()
        test['Content-Type'] = test['Content-Type'].apply(lambda x: str(x).split(';')[0])
        test['Content-Type'][test['Content-Type'] == 'nan'] = ''
        test['Content-Type'][test['Content-Type'] == ''] = 'no_type'
        test['Content-Type'][test['Content-Type'] == 'text/html content-transfer-encoding: 8bit\\r\\n'] = 'text/html'

        # нераспарсеная почта
        test['Text'] = test['Text'].fillna('email')

        test['Subject'] = test['Subject'].fillna('')

        return test, train

    def set_training(self, test, train):
        # test = pd.DataFrame([["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f0", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Easy"],
        #                    ["0e78bba945bc55e5e7027698d4162206f13268f1", "Hard"],
        #                    ], columns=("X-UID", "Label"))
        """## Препроцессинг"""

        def check_sub(x):
            cnt = 0
            x = re.sub(r'[^A-Za-z]', ' ', x)
            for i in x.split():
                if i in wwords:
                    cnt += 1
            if cnt == 0:
                return 0
            if cnt == 1:
                return 0.5
            if cnt > 1:
                return 1

        def subject_cleaner(text):
            text = text.str.strip()
            text = text.str.lower()
            return text

        def body_cleaner(text):
            # text = text.str.replace(r'<[^>]+>', '')
            # text = text.str.replace(r'{[^}]+}', '')
            text = text.str.replace(r'#message', '')
            text = text.str.replace(r'\n{1,}', '')
            text = text.str.replace(r'={1,}', ' ')
            text = text.str.replace(r'-{2,}', ' ')
            text = text.str.replace(r'\*{1,}', ' ')
            text = text.str.replace(r'&nbsp{1,}', ' ')
            text = text.str.replace(r'\t', ' ')
            text = text.str.replace(r'\s{1,}', ' ')
            text = text.str.strip()
            text = text.str.lower()
            return text

        train['Subject'] = subject_cleaner(train['Subject'])
        test['Subject'] = subject_cleaner(test['Subject'])
        train['Text'] = body_cleaner(train['Text'])
        test['Text'] = body_cleaner(test['Text'])

        train['Content-Type'] = train['Content-Type'].apply(lambda x: x.lower().split(';')[0])

        test['Content-Type'] = test['Content-Type'].str.lower().str.strip()
        test['Content-Type'] = test['Content-Type'].apply(lambda x: str(x).split(';')[0])
        test['Content-Type'][test['Content-Type'] == 'nan'] = ''
        test['Content-Type'][test['Content-Type'] == ''] = 'no_type'
        test['Content-Type'][test['Content-Type'] == 'text/html content-transfer-encoding: 8bit\\r\\n'] = 'text/html'

        # нераспарсеная почта
        test['Text'] = test['Text'].fillna('email')

        test['Subject'] = test['Subject'].fillna('')

        """## Токенизация"""

        tfidf_word = TfidfVectorizer(
            # norm='l2',
            analyzer='word',
            ngram_range=(1, 3),
        )
        tfidf_word.fit(train['Text'])
        train_w = tfidf_word.transform(train['Text'])
        test_w = tfidf_word.transform(test['Text'])

        tfidf_char = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 7),
        )
        tfidf_char.fit(train['Text'])
        train_ch = tfidf_char.transform(train['Text'])
        test_ch = tfidf_char.transform(test['Text'])

        train_data = hstack([train_w, train_ch])
        test_data = hstack([test_w, test_ch])

        """## Модель"""

        # Commented out IPython magic to ensure Python compatibility.
        # %%time
        clf = OneClassSVM().fit(train_data)

        # Commented out IPython magic to ensure Python compatibility.
        # %%time
        pred = clf.predict(test_data)

        # я меняю здесь местами 0 и 1 (это сделано специально, чтобы сделать микс с баллами по чеку, см.ниже)
        # если хотите посмотреть скор по тексту, удалите эту запись и добавьте pred[pred == -1] = 0
        pred[pred == 1] = 0
        pred[pred == -1] = 1

        test['Predict_model'] = pred

        """## Дополнительный check для теста

        ### тест отправителя From
        - enron - 0, 
        - внешние - 1
        """

        test['check_email'] = test['From'].str.contains('~@enron') * 1

        """### тест на наличие warning words в теме письма
        - 1 слово - 0.5 баллов, 
        - 2 слова и более - 1 бал, 
        - 0 баллов нет
        """

        wwords = ['payment', 'urgent', 'bank', 'account', 'access', 'block', 'limit', 'confirm', 'important',
                  'password', 'require', 'file', 'download', 'request', 'security', 'validat', 'suspend', 'verificat',
                  'update', 'cash', 'fraud', 'error', 'alert', 'lock', 'card', 'bill', 'official', 'online', 'secure',
                  'profile', 'modif', 'deposit', 'offer', 'verif', 'inquiry', 'free', 'unusual', 'identif',
                  ]

        test['check_subject'] = test['Subject'].apply(check_sub)

        """### тест на наличие warning words в теле письма
        - 1 слово - 0.5 баллов, 
        - 2 слова и более - 1 бал, 
        - 0 баллов нет
        """

        test['check_body'] = test['Text'].apply(check_sub)

        """### тест на наличие http ссылок, за исключением номинального имени сайта, в теле письма
        - есть - 1 бал,
        - нет - 0 баллов
        """

        test['check_text_http'] = test['Text'].str.contains(r'(http|https):\/\/.+?(?=\/)\/\w') * 1

        """### тест столбца Content-Type
        - чистый текст - 0 балов,
        - ссылки, вставки и тд - 3 балла
        """

        test['check_content_type'] = test['Content-Type'].apply(lambda x: 0 if x == 'text/plain' else 3)

        """### сумируем check"""

        test['sum_check'] = test['check_email'] + test['check_subject'] + test['check_body'] + test['check_text_http'] + \
                            test['check_content_type']

        """### Миксуем"""

        test['sum_score'] = test['Predict_model'] + test['sum_check']

        test['Label'] = test['sum_score'].apply(lambda x: 0 if x >= 4 else 1)


        return test

    def update_finish_label(self, df2):
        # raw_data = open("../for_ui.txt").read()
        raw_data = str(df2)
        for row in range(int(len(df2)/300)):
            lbl_row = Label(self.result_frame, relief='groove')
            Label(lbl_row, text=df2['X-UID'][row]).pack(side=LEFT)
            if df2['Label'][row] != "0":
                color = "red"
                self.spam += 1
                point = "Вредоносное письмо"
            else:
                color = "green"
                point = "Невредоносное письмо"
            Label(lbl_row, text=point, background=color, width=20).pack(side=RIGHT)
            lbl_row.pack(fill='x', padx=15, pady=1)
        # self.finish_label.config(text=raw_data)
        # self.canvas.config(text=raw_data)

    def show_statistics(self, df2: pd.DataFrame):
        df2.to_csv("result.csv")
        text = f"Всего писем проверено: {len(df2)}\nИз них вредоносных: {self.spam}\n" \
               f"Файл result.csv успешно сохранён в текущую директорию"
        self.result_text_label.config(text=text)


def main():
    MainRoot()

if __name__ == "__main__":
    main()