import math

class ConvertisseurRadians:

	@staticmethod
	def degres_en_radians(degres):
		radians = degres * math.pi / 180
		return radians

	@staticmethod
	def journee(radian : float):
		# La partie entière de ce nombre contient le nombre d'heures
		nombre_heures_float : float = radian / ((2 * math.pi) / 24)
		# Extraction de la partie entière
		nombre_heures : int = int(nombre_heures_float)
		# On multiplie la valeur après la virgule par (2 * pi) / 24 pour retrouver la partie
		# restante (en radiant) à traiter
		radian_minutes : float = ((2 * math.pi) / 24) * (nombre_heures_float - nombre_heures)
		# on évite la division par 0
		if nombre_heures_float - nombre_heures > 0:
			nombre_minutes : int = int(radian_minutes / ((2 * math.pi) / (24 * 60)))
		else:
			nombre_minutes : int = 0

		return f"{nombre_heures}h{nombre_minutes}m"

	@staticmethod
	def semaine(radian : float):
		# La partie entière de ce nombre contient le nombre de jours
		nombre_jours_float : float = radian / ((2 * math.pi) / 7)
		# Extraction de la partie entière
		nombre_jours = int(nombre_jours_float)
		# On multiplie la valeur après la virgule par (2 * pi) / 7 pour retrouver la partie
		# restante (en radiant) à traiter qui correspond aux heures
		radian_heures : float = ((2 * math.pi) / 7) * (nombre_jours_float - nombre_jours)
		# on évite la division par 0
		if nombre_jours_float - nombre_jours > 0:
			nombre_heures : int = int(radian_heures / ((2 * math.pi) / (7 * 24)))
		else:
			nombre_heures : int = 0

		return f"{nombre_jours}j{nombre_heures}h"

	@staticmethod
	def heure(radian : float):
		# La partie entière de ce nombre contient le nombre de jours
		nombre_minutes_float : float = radian / ((2 * math.pi) / 60)
		# Extraction de la partie entière
		nombre_minutes = int(nombre_minutes_float)
		# On multiplie la valeur après la virgule par (2 * pi) / 7 pour retrouver la partie
		# restante (en radiant) à traiter qui correspond aux heures
		radian_secondes : float = ((2 * math.pi) / 60) * (nombre_minutes_float - nombre_minutes)
		# on évite la division par 0
		if nombre_minutes_float - nombre_minutes > 0:
			nombre_secondes : int = int(radian_secondes / ((2 * math.pi) / (60 * 60)))
		else:
			nombre_secondes : int = 0

		return f"{nombre_minutes}min{nombre_secondes}s"

	@staticmethod
	def angle(radian : float):
		return f"{int(radian / ((2 * math.pi) / 360))}°"
	
	@staticmethod
	def radian_to_unite(radian : float, unite: str):
		match unite:
			case "jour":
				return ConvertisseurRadians.journee(radian)
			case "semaine":
				return ConvertisseurRadians.semaine(radian)
			case "heure":
				return ConvertisseurRadians.heure(radian)
			case "angle":
				return ConvertisseurRadians.angle(radian)
			case _:
				return ""