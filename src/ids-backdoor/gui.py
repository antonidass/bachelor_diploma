# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1034, 548)
        self.choiceFileButton = QtWidgets.QPushButton(Dialog)
        self.choiceFileButton.setGeometry(QtCore.QRect(10, 30, 331, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.choiceFileButton.setFont(font)
        self.choiceFileButton.setObjectName("choiceFileButton")
        self.detectButton = QtWidgets.QPushButton(Dialog)
        self.detectButton.setGeometry(QtCore.QRect(10, 80, 331, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.detectButton.setFont(font)
        self.detectButton.setObjectName("detectButton")
        self.tableWidget = QtWidgets.QTableWidget(Dialog)
        self.tableWidget.setGeometry(QtCore.QRect(360, 30, 671, 441))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.tableWidget.setFont(font)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(25)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(15, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(16, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(17, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(18, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(19, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(20, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(21, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(22, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(23, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(24, item)
        self.histButton = QtWidgets.QPushButton(Dialog)
        self.histButton.setGeometry(QtCore.QRect(10, 130, 331, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.histButton.setFont(font)
        self.histButton.setObjectName("histButton")
        self.allLabel = QtWidgets.QLabel(Dialog)
        self.allLabel.setGeometry(QtCore.QRect(20, 200, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.allLabel.setFont(font)
        self.allLabel.setObjectName("allLabel")
        self.attackLabel = QtWidgets.QLabel(Dialog)
        self.attackLabel.setGeometry(QtCore.QRect(20, 250, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.attackLabel.setFont(font)
        self.attackLabel.setObjectName("attackLabel")
        self.attackRecLabel = QtWidgets.QLabel(Dialog)
        self.attackRecLabel.setGeometry(QtCore.QRect(20, 300, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.attackRecLabel.setFont(font)
        self.attackRecLabel.setObjectName("attackRecLabel")
        self.accuracyLabel = QtWidgets.QLabel(Dialog)
        self.accuracyLabel.setGeometry(QtCore.QRect(20, 350, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.accuracyLabel.setFont(font)
        self.accuracyLabel.setObjectName("accuracyLabel")
        self.allInput = QtWidgets.QPlainTextEdit(Dialog)
        self.allInput.setGeometry(QtCore.QRect(230, 200, 111, 31))
        self.allInput.setReadOnly(True)
        self.allInput.setObjectName("allInput")
        self.attackInput = QtWidgets.QPlainTextEdit(Dialog)
        self.attackInput.setGeometry(QtCore.QRect(230, 250, 111, 31))
        self.attackInput.setReadOnly(True)
        self.attackInput.setObjectName("attackInput")
        self.attackRecInput = QtWidgets.QPlainTextEdit(Dialog)
        self.attackRecInput.setGeometry(QtCore.QRect(230, 300, 111, 31))
        self.attackRecInput.setReadOnly(True)
        self.attackRecInput.setObjectName("attackRecInput")
        self.accuracyInput = QtWidgets.QPlainTextEdit(Dialog)
        self.accuracyInput.setGeometry(QtCore.QRect(230, 350, 111, 31))
        self.accuracyInput.setReadOnly(True)
        self.accuracyInput.setObjectName("accuracyInput")

        self.tableWidget.setColumnWidth(0, 160)
        self.tableWidget.setColumnWidth(2, 100)
        self.tableWidget.setColumnWidth(3, 200)
        self.tableWidget.setColumnWidth(4, 160)
        self.tableWidget.setColumnWidth(5, 160)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate(
            "Dialog", "Распознавание сетевых атак"))
        self.choiceFileButton.setText(_translate("Dialog", "Выбрать файл"))
        self.detectButton.setText(_translate(
            "Dialog", "Обнаружить сетевые атаки"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "Предполагаемая атака"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "Вероятность"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("Dialog", "Атака"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("Dialog", "Продолжительность потока"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("Dialog", "Порт источника"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("Dialog", "Порт назначения"))
        item = self.tableWidget.horizontalHeaderItem(6)
        item.setText(_translate("Dialog", "ID протокола"))
        item = self.tableWidget.horizontalHeaderItem(7)
        item.setText(_translate("Dialog", "Всего пакетов (FWD)"))
        item = self.tableWidget.horizontalHeaderItem(8)
        item.setText(_translate("Dialog", "Всего пакетов (BWD)"))
        item = self.tableWidget.horizontalHeaderItem(9)
        item.setText(_translate("Dialog", "Минимальный размер пакета (FWD)"))
        item = self.tableWidget.horizontalHeaderItem(10)
        item.setText(_translate("Dialog", "Минимальный размер пакета (BWD)"))
        item = self.tableWidget.horizontalHeaderItem(11)
        item.setText(_translate("Dialog", "Максимальный размер пакета (FWD)"))
        item = self.tableWidget.horizontalHeaderItem(12)
        item.setText(_translate("Dialog", "Максимальный размер пакета (BWD)"))
        item = self.tableWidget.horizontalHeaderItem(13)
        item.setText(_translate("Dialog", "Средний размер пакета (FWD)"))
        item = self.tableWidget.horizontalHeaderItem(14)
        item.setText(_translate("Dialog", "Средний размер пакета (BWD)"))
        item = self.tableWidget.horizontalHeaderItem(15)
        item.setText(_translate(
            "Dialog", "Стандартное отклонение пакета (FWD)"))
        item = self.tableWidget.horizontalHeaderItem(16)
        item.setText(_translate(
            "Dialog", "Стандартное отклонение пакета (BWD)"))
        item = self.tableWidget.horizontalHeaderItem(17)
        item.setText(_translate(
            "Dialog", "Минимальное время между двумя пакетами (FWD)"))
        item = self.tableWidget.horizontalHeaderItem(18)
        item.setText(_translate(
            "Dialog", "Минимальное время между двумя пакетами (BWD)"))
        item = self.tableWidget.horizontalHeaderItem(19)
        item.setText(_translate(
            "Dialog", "Среднее время между двумя пакетами (FWD)"))
        item = self.tableWidget.horizontalHeaderItem(20)
        item.setText(_translate(
            "Dialog", "Среднее время между двумя пакетами (BWD)"))
        item = self.tableWidget.horizontalHeaderItem(21)
        item.setText(_translate("Dialog", "Количество пакетов с SYN"))
        item = self.tableWidget.horizontalHeaderItem(22)
        item.setText(_translate("Dialog", "Количество пакетов с FIN"))
        item = self.tableWidget.horizontalHeaderItem(23)
        item.setText(_translate("Dialog", "Количество пакетов с ACK"))
        item = self.tableWidget.horizontalHeaderItem(24)
        item.setText(_translate("Dialog", "Количество пакетов с CWE"))
        self.histButton.setText(_translate("Dialog", "Построить гистограмму"))
        self.allLabel.setText(_translate("Dialog", "Всего записей: "))
        self.attackLabel.setText(_translate("Dialog", "Всего атак: "))
        self.attackRecLabel.setText(_translate(
            "Dialog", "Всего атак распознано: "))
        self.accuracyLabel.setText(_translate(
            "Dialog", "Точность распознавания:"))