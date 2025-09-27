-- --------------------------------------------------------
-- Hôte:                         127.0.0.1
-- Version du serveur:           8.0.42 - MySQL Community Server - GPL
-- SE du serveur:                Win64
-- HeidiSQL Version:             12.10.0.7000
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

-- Listage de la structure de table mkmpy2. logsteps
CREATE TABLE IF NOT EXISTS `logsteps` (
  `id` int NOT NULL AUTO_INCREMENT,
  `step` varchar(15) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=91 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Listage des données de la table mkmpy2.logsteps : ~3 rows (environ)
INSERT IGNORE INTO `logsteps` (`id`, `step`) VALUES
	(10, 'ongoing'),
	(50, 'finished'),
	(90, 'too early');

-- Listage de la structure de table mkmpy2. logs_oracle
CREATE TABLE IF NOT EXISTS `logs_oracle` (
  `id` int NOT NULL AUTO_INCREMENT,
  `date` datetime DEFAULT CURRENT_TIMESTAMP,
  `idStep` int DEFAULT NULL,
  `idTask` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `IDX_99CE53ED3044A0A2` (`idStep`),
  KEY `IDX_99CE53ED218385BB` (`idTask`),
  CONSTRAINT `FK_99CE53ED218385BB` FOREIGN KEY (`idTask`) REFERENCES `taskstypes` (`id`),
  CONSTRAINT `FK_99CE53ED3044A0A2` FOREIGN KEY (`idStep`) REFERENCES `logsteps` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Listage des données de la table mkmpy2.logs_oracle : ~0 rows (environ)

-- Listage de la structure de table mkmpy2. prices_predict
CREATE TABLE IF NOT EXISTS `prices_predict` (
  `id` int NOT NULL AUTO_INCREMENT,
  `date_data` date DEFAULT NULL,
  `avg1` decimal(10,2) DEFAULT NULL,
  `avg1_foil` decimal(10,2) DEFAULT NULL,
  `idProduct` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `IDX_FEFDEDC5C3F36F5F` (`idProduct`),
  CONSTRAINT `FK_FEFDEDC5C3F36F5F` FOREIGN KEY (`idProduct`) REFERENCES `products` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Listage des données de la table mkmpy2.prices_predict : ~0 rows (environ)

-- Listage de la structure de table mkmpy2. taskstypes
CREATE TABLE IF NOT EXISTS `taskstypes` (
  `id` int NOT NULL AUTO_INCREMENT,
  `task` varchar(15) COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=21 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Listage des données de la table mkmpy2.taskstypes : ~2 rows (environ)
INSERT IGNORE INTO `taskstypes` (`id`, `task`) VALUES
	(10, 'learn'),
	(20, 'predict');

/*!40103 SET TIME_ZONE=IFNULL(@OLD_TIME_ZONE, 'system') */;
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IFNULL(@OLD_FOREIGN_KEY_CHECKS, 1) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40111 SET SQL_NOTES=IFNULL(@OLD_SQL_NOTES, 1) */;
