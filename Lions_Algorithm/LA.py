import random
import numpy as np
import sys
from K_clustering import *

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

def f1(x):
	return sum(pow(x, 2))

class LA:

	def __init__(self):
		self.f = f1
		self.cluster = K_Cluster(2)
		self.dimension = 5
		self.age_mat = 3
		self.B_strength = 5
		self.prob_cross = [0.4, 0.6]
		self.prob_mut = 0.8
		self.new_female_th = 1000
		self.min_val = [-10*pow(10,3), -10*pow(10,3), -10*pow(10,3), -10*pow(10,3), -10*pow(10,3)]
		self.max_val = [ 10*pow(10,3),  10*pow(10,3),  10*pow(10,3),  10*pow(10,3),  10*pow(10,3)]
		self.generations = 1000

	def CalcFitness(self, individuos):
		return np.apply_along_axis(self.f, 1, individuos)

	def CreateRandomLions(self, size):
		return np.array([[np.random.randint(self.min_val[j], self.max_val[j]) for j in range(self.dimension)] for i in range(size)]).astype(np.int64)

	def InitPob(self):
		self.male = self.CreateRandomLions(1)[0]
		self.male_fit = self.CalcFitness([self.male])[0]
		self.female = self.CreateRandomLions(1)[0]
		self.female_fit = self.CalcFitness([self.female])[0]
		self.B_count = 0

	def Crossover(self, l1, l2, prob):
		cross = random.random()
		if cross < prob:
			cut = np.random.randint(0, len(l1))
			cub1 = np.concatenate((l1[:cut], l2[cut:]))
			cub2 = np.concatenate((l2[:cut], l1[cut:]))
		else:
			cub1 = np.copy(l1)
			cub2 = np.copy(l2)
		return cub1, cub2

	def Mutation(self, cub, prob):
		mut = random.random()
		if mut < prob:
			pos = np.random.randint(0, len(cub))
			cub[pos] = random.uniform(self.min_val[pos], self.max_val[pos])
		return cub

	def GenderGrouping(self, cubs):
		groups = self.cluster.GetGroups(cubs, np.concatenate(([self.male],[self.female])))
		self.m_cubs = groups[0]
		self.m_cubs_fit = self.CalcFitness(self.m_cubs)
		self.f_cubs = groups[1]
		self.f_cubs_fit = self.CalcFitness(self.f_cubs)

	def KillWeakCubs(self):
		m_sort = self.m_cubs_fit.argsort()
		f_sort = self.f_cubs_fit.argsort()
		self.m_cubs = self.m_cubs[m_sort]
		self.m_cubs_fit = self.m_cubs_fit[m_sort]
		self.f_cubs = self.f_cubs[f_sort]
		self.f_cubs_fit = self.f_cubs_fit[f_sort]

		diff = abs(len(self.m_cubs) - len(self.f_cubs))
		if diff == 0:
			return
		if len(self.m_cubs) < len(self.f_cubs):
			self.f_cubs = self.f_cubs[:-diff]
			self.f_cubs_fit = self.f_cubs_fit[:-diff]
		else:
			self.m_cubs = self.m_cubs[:-diff]
			self.m_cubs_fit = self.m_cubs_fit[:-diff]

	def Mating(self):
		#Crossover
		cub1, cub2 = self.Crossover(self.male, self.female, self.prob_cross[0])
		cub3, cub4 = self.Crossover(self.male, self.female, self.prob_cross[1])
		cubs = np.array([cub1, cub2, cub3, cub4])

		#Mutation
		for i in range(len(cubs)):
			mut = self.Mutation(np.copy(cubs[i]), self.prob_mut)
			cubs = np.concatenate((cubs, [mut]))

		#Gender Grouping
		self.GenderGrouping(cubs)

		#Kill Weaks
		self.KillWeakCubs()

		#Update Pride
		self.age_cubs = 0

	def GetPrideFitness(self):
		cubs_fit = 0
		for i in range(len(self.m_cubs)):
			cubs_fit += self.m_cubs_fit[i] + self.f_cubs_fit[i]
		cubs_fit /= len(self.m_cubs)
		return (self.male_fit + self.female_fit + self.age_mat / (self.age_cubs + 1) * cubs_fit) / \
			   (2 * (1 + len(self.m_cubs)))

	def TerritorialDefense(self):
		while self.age_cubs < self.age_mat:
			nomad = self.CreateRandomLions(1)[0]
			nomad_fit = self.CalcFitness([nomad])[0]
			if nomad_fit < self.male_fit:
				if nomad_fit < self.GetPrideFitness():
					self.male = nomad
					self.male_fit = nomad_fit
					self.Mating()
			else:
				self.age_cubs += 1

	def TerritorialTakeover(self):
		#Select best male
		best_m_cub_arg = 0
		if self.m_cubs_fit[best_m_cub_arg] < self.male_fit:
			self.male = self.m_cubs[best_m_cub_arg]
			self.male_fit = self.m_cubs_fit[best_m_cub_arg]

		#Select best female
		best_f_cub_arg = 0
		while (self.f_cubs[best_f_cub_arg] == self.male).all() and best_f_cub_arg < len(self.f_cubs)-1:
			best_f_cub_arg += 1
		if self.f_cubs_fit[best_f_cub_arg] < self.female_fit:
			self.female = self.f_cubs[best_f_cub_arg]
			self.female_fit = self.f_cubs_fit[best_f_cub_arg]
			self.B_count = 0
		else:
			self.B_count += 1

		#If female has had enough breedings, generate a new one (better)
		if self.B_count > self.B_strength:
			new_female_fit = np.inf
			new_female_count = 0
			while new_female_fit >= self.female_fit and new_female_count < self.new_female_th:
				new_female = self.Mutation(np.copy(self.female), 1.0)
				new_female_fit = self.CalcFitness([new_female])[0]
				new_female_count += 1
			self.female = new_female
			self.female_fit = new_female_fit
			self.B_count = 0

	def GetBestLion(self):
		if self.male_fit < self.female_fit:
			return self.male
		else:
			return self.female

	def PrintMaleFemale(self):
		print ("Male: ", self.male, '----------', self.male_fit)
		print ("Female: ", self.female, '----------', self.female_fit)

	def PrintLions(self, lions, fitness):
		for i in range(len(lions)):
			print (lions[i], '----------', fitness[i])

	def Run(self):
		self.InitPob()

		print ("Initial Pride: ")
		self.PrintMaleFemale()
		print ("------------------------------------------------------------------------")
		for i in range(self.generations):

			print("ITERATION %d:" % (i))

			self.PrintMaleFemale()

			print("------------------- MATING -------------------")
			self.Mating()
			print("Male cubs: ")
			self.PrintLions(self.m_cubs, self.m_cubs_fit)
			print("Female cubs: ")
			self.PrintLions(self.f_cubs, self.f_cubs_fit)

			print("------------ TERRITORIAL DEFENSE -------------")
			self.TerritorialDefense()
			self.PrintMaleFemale()
			print("Male cubs: ")
			self.PrintLions(self.m_cubs, self.m_cubs_fit)
			print("Female cubs: ")
			self.PrintLions(self.f_cubs, self.f_cubs_fit)

			print("----------- TERRITORIAL TAKEOVER -------------")
			self.TerritorialTakeover()
			self.PrintMaleFemale()
			print("Breed count: ", self.B_count)

			print("Best Lion (local):")
			self.PrintLions([self.GetBestLion()], self.CalcFitness([self.GetBestLion()]))
			print("------------------------------------------------------------------------ END ITERATION %d" % (i))

		print("************************************************************************")
		print("FINAL RESULT (%d generations):"%(self.generations))
		self.PrintMaleFemale()
		print("Male cubs: ")
		self.PrintLions(self.m_cubs, self.m_cubs_fit)
		print("Female cubs: ")
		self.PrintLions(self.f_cubs, self.f_cubs_fit)
		print("Best Lion:")
		self.PrintLions([self.GetBestLion()], self.CalcFitness([self.GetBestLion()]))
		print("************************************************************************")

L = LA()
L.Run()

sys.stdout = orig_stdout
f.close()