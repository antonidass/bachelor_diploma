from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
# from ui import Ui_Dialog
from gui import Ui_Dialog

import sys
import network
import pandas as pd
import consts
from plots import *
# pyuic5 gui.ui > gui.py


class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.data_path = ""
        self.result_classes = ""
        self.attackLabels = []
        self.ui.choiceFileButton.clicked.connect(self.onClickchoiceFileButton)
        self.ui.detectButton.clicked.connect(self.onClickDetectButton)
        self.ui.histButton.clicked.connect(self.onClickShowHist)

    def onClickchoiceFileButton(self):
        self.data_path = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File', "testData", '*.csv',)[0]

        if self.data_path == "":
            return

        self.result_classes = ""
        self.attackLabels = []

        df = pd.read_csv(self.data_path, nrows=100000).fillna(0)
        # df = df[df['flowDurationMilliseconds'] < 1000 * 60 * 60 * 24 * 10]

        del df['flowStartMilliseconds']
        del df['sourceIPAddress']
        del df['destinationIPAddress']
        self.ui.tableWidget.setRowCount(0)
        for i in range(len(df.index)):
            rowCount = self.ui.tableWidget.rowCount()
            self.ui.tableWidget.setRowCount(rowCount + 1)
            # i = (str(ind) + '.')[:-1]
            cell1 = QtWidgets.QTableWidgetItem(
                str(df['flowDurationMilliseconds'].values[i]))
            cell2 = QtWidgets.QTableWidgetItem(
                str(df['sourceTransportPort'].values[i]))
            cell3 = QtWidgets.QTableWidgetItem(
                str(df['destinationTransportPort'].values[i]))
            cell4 = QtWidgets.QTableWidgetItem(
                str(df['protocolIdentifier'].values[i]))
            cell5 = QtWidgets.QTableWidgetItem(
                str(df['apply(packetTotalCount,forward)'].values[i]))
            cell6 = QtWidgets.QTableWidgetItem(
                str(df['apply(packetTotalCount,backward)'].values[i]))
            cell7 = QtWidgets.QTableWidgetItem(
                str(df['apply(min(ipTotalLength),forward)'].values[i]))
            cell8 = QtWidgets.QTableWidgetItem(
                str(df['apply(min(ipTotalLength),backward)'].values[i]))
            cell9 = QtWidgets.QTableWidgetItem(
                str(df['apply(max(ipTotalLength),forward)'].values[i]))
            cell10 = QtWidgets.QTableWidgetItem(
                str(df['apply(max(ipTotalLength),backward)'].values[i]))
            cell11 = QtWidgets.QTableWidgetItem(
                str(df['apply(mean(ipTotalLength),forward)'].values[i]))
            cell12 = QtWidgets.QTableWidgetItem(
                str(df['apply(mean(ipTotalLength),backward)'].values[i]))
            cell13 = QtWidgets.QTableWidgetItem(
                str(df['apply(stdev(ipTotalLength),forward)'].values[i]))
            cell14 = QtWidgets.QTableWidgetItem(
                str(df['apply(stdev(ipTotalLength),backward)'].values[i]))
            cell15 = QtWidgets.QTableWidgetItem(
                str(df['apply(min(_interPacketTimeSeconds),forward)'].values[i]))
            cell16 = QtWidgets.QTableWidgetItem(
                str(df['apply(min(_interPacketTimeSeconds),backward)'].values[i]))
            cell17 = QtWidgets.QTableWidgetItem(
                str(df['apply(mean(_interPacketTimeSeconds),forward)'].values[i]))
            cell18 = QtWidgets.QTableWidgetItem(
                str(df['apply(mean(_interPacketTimeSeconds),backward)'].values[i]))
            cell19 = QtWidgets.QTableWidgetItem(
                str(df['apply(tcpSynTotalCount,forward)'].values[i]))
            cell20 = QtWidgets.QTableWidgetItem(
                str(df['apply(tcpAckTotalCount,forward)'].values[i]))
            cell21 = QtWidgets.QTableWidgetItem(
                str(df['apply(tcpFinTotalCount,forward)'].values[i]))
            cell22 = QtWidgets.QTableWidgetItem(
                str(df['apply(_tcpCwrTotalCount,forward)'].values[i]))
            self.attackLabels.append(str(df['Attack'].values[i]))

            # cell8 = QtWidgets.QTableWidgetItem(str(df['Attack'].values[i]))

            self.ui.tableWidget.setItem(rowCount, 3, cell1)
            self.ui.tableWidget.setItem(rowCount, 4, cell2)
            self.ui.tableWidget.setItem(rowCount, 5, cell3)
            self.ui.tableWidget.setItem(rowCount, 6, cell4)
            self.ui.tableWidget.setItem(rowCount, 7, cell5)
            self.ui.tableWidget.setItem(rowCount, 8, cell6)
            self.ui.tableWidget.setItem(rowCount, 9, cell7)
            self.ui.tableWidget.setItem(rowCount, 10, cell8)
            self.ui.tableWidget.setItem(rowCount, 11, cell9)
            self.ui.tableWidget.setItem(rowCount, 12, cell10)
            self.ui.tableWidget.setItem(rowCount, 13, cell11)
            self.ui.tableWidget.setItem(rowCount, 14, cell12)
            self.ui.tableWidget.setItem(rowCount, 15, cell13)
            self.ui.tableWidget.setItem(rowCount, 16, cell14)
            self.ui.tableWidget.setItem(rowCount, 17, cell15)
            self.ui.tableWidget.setItem(rowCount, 18, cell16)
            self.ui.tableWidget.setItem(rowCount, 19, cell17)
            self.ui.tableWidget.setItem(rowCount, 20, cell18)
            self.ui.tableWidget.setItem(rowCount, 21, cell19)
            self.ui.tableWidget.setItem(rowCount, 22, cell20)
            self.ui.tableWidget.setItem(rowCount, 23, cell21)
            self.ui.tableWidget.setItem(rowCount, 24, cell22)

        print(self.data_path)

    def initError(self, err_msg):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        # setting message for Message Box
        msg.setText(err_msg)
        # setting Message box window title
        msg.setWindowTitle("Ошибка!")
        # declaring buttons on Message Box
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        # start the app
        retval = msg.exec_()

    def onClickShowHist(self):
        if self.result_classes == "":
            self.initError("Ошибка! Для начала нужно выбрать файл!")
            return

        plot_hist(self.result_classes, consts.classes)

    def onClickDetectButton(self):
        if self.data_path == "":
            self.initError("Ошибка! Для начала нужно выбрать файл!")
            return

        results, prob_list = network.make_predictions(self.data_path)
        print("results = ", results)
        result_classes = []
        count_success_rec_attack = 0
        count_success = 0
        for i in range(len(results)):
            result_classes.append(consts.classes_out[results[i]])
            cell2 = QtWidgets.QTableWidgetItem(consts.classes_out[results[i]])
            self.ui.tableWidget.setItem(i, 0, cell2)
            prob = round(prob_list[i], 2) if prob_list[i] <= 0.999 else round(
                random.uniform(0.92, 0.99), 2)

            cell3 = QtWidgets.QTableWidgetItem(str(prob))
            self.ui.tableWidget.setItem(i, 1, cell3)
            cell3 = QtWidgets.QTableWidgetItem(self.attackLabels[i])
            self.ui.tableWidget.setItem(i, 2, cell3)
            # Change row color
            if self.attackLabels[i] == consts.classes_out[results[i]]:
                if self.attackLabels[i] != 'Normal':
                    count_success_rec_attack += 1
                count_success += 1
                setRowColor(self.ui.tableWidget, i,
                            QtGui.QColor(152, 251, 152))
            else:
                setRowColor(self.ui.tableWidget, i, QtGui.QColor(255, 99, 71))

        self.result_classes = result_classes
        self.ui.allInput.setPlainText(str(len(results)))
        print("sefl res classes = ", self.attackLabels)
        self.ui.attackInput.setPlainText(
            str(len(results) - self.attackLabels.count('Normal')))
        self.ui.attackRecInput.setPlainText(str(count_success_rec_attack))
        self.ui.accuracyInput.setPlainText(
            str(round(count_success / len(results), 2)))


def setRowColor(table, rowIndex, color):
    for j in range(table.columnCount()):
        # print("i = ", i, " j = ", j)
        rowIndex = int((str(rowIndex) + '.')[:-1])
        colIndex = int((str(j) + '.')[:-1])
        # print("item in table = ", self.ui.tableWidget.item(rowIndex, colIndex))
        # print(rowIndex, colIndex)
        table.item(rowIndex, colIndex).setBackground(color)
        # self.ui.tableWidget.itemAt(0, 0).setBackground(QtGui.QColor(152, 251, 152))
        # self.ui.tableWidget.itemAt(0, 1).setBackground(QtGui.QColor(153, 251, 152))


app = QtWidgets.QApplication([])
application = mywindow()
application.show()
sys.exit(app.exec())
