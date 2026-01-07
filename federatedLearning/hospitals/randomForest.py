from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class RandomForest:

    def __init__(self, df):
        self.df = df

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def preprocessing(self):
                
        self.df.columns = self.df.columns.str.replace('#', '').str.strip().str.lower()

        self.df.rename(columns={'heart_rate': 'bpm', 
                        'activityid': 'target'}, 
                inplace=True)

        self.df = self.df[self.df['target'] != 'transient activities'].copy()
    

    def encoding(self):
        
        le = LabelEncoder()

        self.df['target'] = le.fit_transform(self.df['target']) 

        self.df['target'] = self.df['target'].apply(lambda x: 0 if x <= 2 else 1)

        self.df.dropna(inplace=True)


    def createModel(self):

        window_size = 5

        self.df["rolling_mean"] = self.df["bpm"].rolling(window=window_size).mean()
        self.df["rolling_std"] = self.df["bpm"].rolling(window=window_size).std()
        self.df["diff"] = self.df["bpm"].diff()

        self.df.dropna(inplace=True)

        features = ["bpm", "rolling_mean", "rolling_std", "diff"]
        X = self.df[features]
        y = self.df["target"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        self.model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42) 


    def results(self):

        y_pred = self.model.predict(self.X_test)
        print(
            classification_report(
                self.y_test,
                y_pred,
                target_names=["Repouso", "Atividade/Anormal"]
            )
        )