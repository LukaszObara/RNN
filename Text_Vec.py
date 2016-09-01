import numpy as np

class Text_Vec(object):
	def __init__(self, location='C:\\Users\\Lukasz Obara\\OneDrive'
				 +'\\Documents\\Machine Learning\\Text Files'):
		self.location = location

	def convert_text(self, file_name, save=False):
		file = self.location + file_name
		text_file = open(file)
		text = text_file.read()
		text_file.close()

		text_list = [np.array([[0] for i in range(127)]) 
					 for j in range(len(text))]

		for i in range(len(text_list)-1):
			ascii_num = ord(text[i]) 
			text_list[i][ascii_num] = 1

		if save:
			np.savetxt(self.location + '\\test.csv', text_list, 
					   fmt='%1d', delimiter=',')

		return text_list

if __name__ == '__main__':
	pass
