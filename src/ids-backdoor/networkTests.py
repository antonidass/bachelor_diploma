import unittest
import pandas as pd
import network
import consts


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.csv_path = "testData/TEST.csv"
        df = pd.read_csv(self.csv_path, nrows=100000).fillna(0)
        del df['flowStartMilliseconds']
        del df['sourceIPAddress']
        del df['destinationIPAddress']

        self.attackLabels = []

        for i in range(len(df.index)):
            self.attackLabels.append(
                consts.classes.index(str(df['Attack'].values[i])))

        self.results, self.prob_list = network.make_predictions(self.csv_path)

    def testAttackNotDetectedOneAttack(self):
        self.assertNotEqual(
            self.results[8], self.attackLabels
        )

    def testAttackDetectedOther(self):
        self.assertEqual(
            self.results[0:8] + self.results[9:],
            self.attackLabels[0:8] + self.attackLabels[9:]
        )

    def testLoitAttackDetected(self):
        self.assertEqual(
            consts.classes[self.results[0]],
            "DDoS:LOIT"
        )

    def testInfiltrationAttackDetected(self):
        self.assertEqual(
            consts.classes[self.results[-1]],
            "Infiltration:Dropbox download - (Portscan + Nmap) from victim"
        )
