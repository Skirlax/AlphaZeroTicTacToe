-- MySQL dump 10.13  Distrib 8.0.34, for Linux (x86_64)
--
-- Host: localhost    Database: alpha_zero
-- ------------------------------------------------------
-- Server version	8.0.34-0ubuntu0.20.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `alembic_version`
--

DROP TABLE IF EXISTS `alembic_version`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `alembic_version` (
  `version_num` varchar(32) NOT NULL,
  PRIMARY KEY (`version_num`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `alembic_version`
--

LOCK TABLES `alembic_version` WRITE;
/*!40000 ALTER TABLE `alembic_version` DISABLE KEYS */;
INSERT INTO `alembic_version` VALUES ('v3.2.0.a');
/*!40000 ALTER TABLE `alembic_version` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `studies`
--

DROP TABLE IF EXISTS `studies`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `studies` (
  `study_id` int NOT NULL AUTO_INCREMENT,
  `study_name` varchar(512) NOT NULL,
  PRIMARY KEY (`study_id`),
  UNIQUE KEY `ix_studies_study_name` (`study_name`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `studies`
--

LOCK TABLES `studies` WRITE;
/*!40000 ALTER TABLE `studies` DISABLE KEYS */;
INSERT INTO `studies` VALUES (1,'alpha_zero');
/*!40000 ALTER TABLE `studies` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `study_directions`
--

DROP TABLE IF EXISTS `study_directions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `study_directions` (
  `study_direction_id` int NOT NULL AUTO_INCREMENT,
  `direction` enum('NOT_SET','MINIMIZE','MAXIMIZE') NOT NULL,
  `study_id` int NOT NULL,
  `objective` int NOT NULL,
  PRIMARY KEY (`study_direction_id`),
  UNIQUE KEY `study_id` (`study_id`,`objective`),
  CONSTRAINT `study_directions_ibfk_1` FOREIGN KEY (`study_id`) REFERENCES `studies` (`study_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `study_directions`
--

LOCK TABLES `study_directions` WRITE;
/*!40000 ALTER TABLE `study_directions` DISABLE KEYS */;
INSERT INTO `study_directions` VALUES (1,'MAXIMIZE',1,0);
/*!40000 ALTER TABLE `study_directions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `study_system_attributes`
--

DROP TABLE IF EXISTS `study_system_attributes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `study_system_attributes` (
  `study_system_attribute_id` int NOT NULL AUTO_INCREMENT,
  `study_id` int DEFAULT NULL,
  `key` varchar(512) DEFAULT NULL,
  `value_json` text,
  PRIMARY KEY (`study_system_attribute_id`),
  UNIQUE KEY `study_id` (`study_id`,`key`),
  CONSTRAINT `study_system_attributes_ibfk_1` FOREIGN KEY (`study_id`) REFERENCES `studies` (`study_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `study_system_attributes`
--

LOCK TABLES `study_system_attributes` WRITE;
/*!40000 ALTER TABLE `study_system_attributes` DISABLE KEYS */;
/*!40000 ALTER TABLE `study_system_attributes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `study_user_attributes`
--

DROP TABLE IF EXISTS `study_user_attributes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `study_user_attributes` (
  `study_user_attribute_id` int NOT NULL AUTO_INCREMENT,
  `study_id` int DEFAULT NULL,
  `key` varchar(512) DEFAULT NULL,
  `value_json` text,
  PRIMARY KEY (`study_user_attribute_id`),
  UNIQUE KEY `study_id` (`study_id`,`key`),
  CONSTRAINT `study_user_attributes_ibfk_1` FOREIGN KEY (`study_id`) REFERENCES `studies` (`study_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `study_user_attributes`
--

LOCK TABLES `study_user_attributes` WRITE;
/*!40000 ALTER TABLE `study_user_attributes` DISABLE KEYS */;
/*!40000 ALTER TABLE `study_user_attributes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trial_heartbeats`
--

DROP TABLE IF EXISTS `trial_heartbeats`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trial_heartbeats` (
  `trial_heartbeat_id` int NOT NULL AUTO_INCREMENT,
  `trial_id` int NOT NULL,
  `heartbeat` datetime NOT NULL,
  PRIMARY KEY (`trial_heartbeat_id`),
  UNIQUE KEY `trial_id` (`trial_id`),
  CONSTRAINT `trial_heartbeats_ibfk_1` FOREIGN KEY (`trial_id`) REFERENCES `trials` (`trial_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trial_heartbeats`
--

LOCK TABLES `trial_heartbeats` WRITE;
/*!40000 ALTER TABLE `trial_heartbeats` DISABLE KEYS */;
/*!40000 ALTER TABLE `trial_heartbeats` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trial_intermediate_values`
--

DROP TABLE IF EXISTS `trial_intermediate_values`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trial_intermediate_values` (
  `trial_intermediate_value_id` int NOT NULL AUTO_INCREMENT,
  `trial_id` int NOT NULL,
  `step` int NOT NULL,
  `intermediate_value` double DEFAULT NULL,
  `intermediate_value_type` enum('FINITE','INF_POS','INF_NEG','NAN') NOT NULL,
  PRIMARY KEY (`trial_intermediate_value_id`),
  UNIQUE KEY `trial_id` (`trial_id`,`step`),
  CONSTRAINT `trial_intermediate_values_ibfk_1` FOREIGN KEY (`trial_id`) REFERENCES `trials` (`trial_id`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trial_intermediate_values`
--

LOCK TABLES `trial_intermediate_values` WRITE;
/*!40000 ALTER TABLE `trial_intermediate_values` DISABLE KEYS */;
INSERT INTO `trial_intermediate_values` VALUES (1,1,0,0.925,'FINITE'),(2,6,5,0.7649999999999999,'FINITE'),(3,4,3,0.7450000000000001,'FINITE'),(4,5,4,0.78,'FINITE'),(5,8,7,0.835,'FINITE'),(6,7,6,0.96,'FINITE'),(7,9,8,0.7899999999999999,'FINITE'),(8,11,10,0.765,'FINITE'),(9,10,9,0.8400000000000001,'FINITE'),(10,12,11,0.845,'FINITE'),(11,13,12,0.8300000000000001,'FINITE'),(12,16,15,0.9200000000000002,'FINITE'),(13,14,13,0.8550000000000001,'FINITE');
/*!40000 ALTER TABLE `trial_intermediate_values` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trial_params`
--

DROP TABLE IF EXISTS `trial_params`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trial_params` (
  `param_id` int NOT NULL AUTO_INCREMENT,
  `trial_id` int DEFAULT NULL,
  `param_name` varchar(512) DEFAULT NULL,
  `param_value` double DEFAULT NULL,
  `distribution_json` text,
  PRIMARY KEY (`param_id`),
  UNIQUE KEY `trial_id` (`trial_id`,`param_name`),
  CONSTRAINT `trial_params_ibfk_1` FOREIGN KEY (`trial_id`) REFERENCES `trials` (`trial_id`)
) ENGINE=InnoDB AUTO_INCREMENT=145 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trial_params`
--

LOCK TABLES `trial_params` WRITE;
/*!40000 ALTER TABLE `trial_params` DISABLE KEYS */;
INSERT INTO `trial_params` VALUES (1,1,'num_mc_simulations',226,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(2,1,'num_self_play_games',174,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(3,2,'num_mc_simulations',303,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(4,1,'num_epochs',163,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(5,2,'num_self_play_games',139,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(6,3,'num_mc_simulations',367,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(7,1,'lr',0.00027465561237168646,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(8,2,'num_epochs',335,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(9,3,'num_self_play_games',118,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(10,4,'num_mc_simulations',856,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(11,2,'lr',0.00016233529358623705,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(12,1,'temp',0.6746280345746684,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(13,3,'num_epochs',286,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(14,2,'temp',0.8964080561617243,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(15,4,'num_self_play_games',119,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(16,1,'arena_temp',0.10971154513352581,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(17,5,'num_mc_simulations',682,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(18,3,'lr',0.0007153424859697276,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(19,1,'cpuct',3.752413262959074,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(20,5,'num_self_play_games',134,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(21,2,'arena_temp',0.4532580660954949,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(22,4,'num_epochs',280,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(23,3,'temp',0.8235056182973706,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(24,1,'log_epsilon',0.000000020662108040771462,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(25,4,'lr',0.004360828332527576,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(26,2,'cpuct',1.9123285102051635,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(27,3,'arena_temp',0.26847412835233103,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(28,5,'num_epochs',365,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(29,4,'temp',1.4584107009295422,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(30,3,'cpuct',4.750244077698926,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(31,2,'log_epsilon',0.0000000007022400738956882,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(32,5,'lr',0.009073289006381435,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(33,4,'arena_temp',0.07606587341945516,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(34,3,'log_epsilon',0.00000003683725557182956,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(35,5,'temp',0.5707000199206913,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(36,4,'cpuct',3.41025232022506,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(37,4,'log_epsilon',0.0000000005699827228292922,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(38,5,'arena_temp',0.4233992870668901,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(39,5,'cpuct',0.5651139712613882,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(40,5,'log_epsilon',0.0000000013783112373608032,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(41,6,'num_mc_simulations',173,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(42,6,'num_self_play_games',60,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(43,6,'num_epochs',293,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(44,6,'lr',0.0001160809826321313,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(45,6,'temp',1.3809426980633797,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(46,6,'arena_temp',0.21296660783113427,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(47,6,'cpuct',3.8756232979089886,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(48,6,'log_epsilon',0.00000000544770006085397,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(49,7,'num_mc_simulations',1317,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(50,7,'num_self_play_games',142,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(51,7,'num_epochs',320,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(52,7,'lr',0.00032485504583772953,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(53,7,'temp',0.8243290871234756,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(54,7,'arena_temp',0.04139160592420218,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(55,7,'cpuct',1.7409658274805089,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(56,7,'log_epsilon',0.000000014165210108199043,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(57,8,'num_mc_simulations',836,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(58,8,'num_self_play_games',135,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(59,8,'num_epochs',192,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(60,8,'lr',0.0002751047510395801,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(61,8,'temp',1.214188187387272,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(62,8,'arena_temp',0.24785478379259956,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(63,8,'cpuct',1.3985761691462049,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(64,8,'log_epsilon',0.00000007411098796567593,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(65,9,'num_mc_simulations',878,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(66,9,'num_self_play_games',163,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(67,9,'num_epochs',269,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(68,9,'lr',0.002895915369834172,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(69,9,'temp',1.3318359497950356,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(70,9,'arena_temp',0.055287265482688815,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(71,9,'cpuct',2.6185697273699207,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(72,9,'log_epsilon',0.0000000001861732115666699,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(73,10,'num_mc_simulations',1064,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(74,10,'num_self_play_games',119,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(75,10,'num_epochs',397,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(76,10,'lr',0.0001509089244344365,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(77,10,'temp',0.8692770259500033,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(78,10,'arena_temp',0.4338999075982349,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(79,10,'cpuct',4.687830375170339,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(80,10,'log_epsilon',0.000000004710733106679701,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(81,11,'num_mc_simulations',1371,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(82,11,'num_self_play_games',103,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(83,11,'num_epochs',157,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(84,11,'lr',0.0024008989411230374,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(85,11,'temp',1.3342683093145322,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(86,11,'arena_temp',0.07093136144738135,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(87,11,'cpuct',2.5187358335388406,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(88,11,'log_epsilon',0.00000003327767578686854,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(89,12,'num_mc_simulations',1411,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(90,12,'num_self_play_games',80,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(91,12,'num_epochs',155,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(92,12,'lr',0.00024139664084019236,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(93,12,'temp',0.5026010412336082,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(94,12,'arena_temp',0.2507267172295029,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(95,12,'cpuct',4.769556262440675,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(96,12,'log_epsilon',0.0000000001297547655750155,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(97,13,'num_mc_simulations',627,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(98,13,'num_self_play_games',120,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(99,13,'num_epochs',281,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(100,13,'lr',0.006730382641311698,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(101,13,'temp',1.2009737100639288,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(102,13,'arena_temp',0.08741391246618649,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(103,13,'cpuct',2.2201429539776676,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(104,13,'log_epsilon',0.00000001192623466047923,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(105,14,'num_mc_simulations',1477,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(106,14,'num_self_play_games',82,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(107,14,'num_epochs',125,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(108,14,'lr',0.0002923620255234689,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(109,14,'temp',0.6152687434434211,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(110,14,'arena_temp',0.20473513817508812,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(111,14,'cpuct',1.6362082737524841,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(112,14,'log_epsilon',0.000000005147659536144295,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(113,15,'num_mc_simulations',1576,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(114,15,'num_self_play_games',196,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(115,15,'num_epochs',331,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(116,15,'lr',0.0008327254333499607,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(117,15,'temp',1.0317282311317828,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(118,15,'arena_temp',0.02281733366420299,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(119,15,'cpuct',1.7977806076662097,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(120,15,'log_epsilon',0.00000000961764825405259,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(121,16,'num_mc_simulations',197,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(122,16,'num_self_play_games',200,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(123,16,'num_epochs',209,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(124,16,'lr',0.0005697313236120295,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(125,16,'temp',0.7833402160710213,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(126,16,'arena_temp',0.01910802277731641,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(127,16,'cpuct',3.3850390573455234,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(128,16,'log_epsilon',0.000000016475137268509303,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(129,17,'num_mc_simulations',1549,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(130,17,'num_self_play_games',171,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(131,17,'num_epochs',105,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(132,17,'lr',0.0006322509082194958,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(133,17,'temp',0.796217822126039,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(134,17,'arena_temp',0.15573681840576273,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(135,17,'cpuct',1.8707181802795598,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(136,17,'log_epsilon',0.00000009789419862883908,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}'),(137,18,'num_mc_simulations',369,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 60, \"high\": 1600}}'),(138,18,'num_self_play_games',169,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 50, \"high\": 200}}'),(139,18,'num_epochs',335,'{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 100, \"high\": 400}}'),(140,18,'lr',0.0008578431521516828,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.0001, \"high\": 0.01, \"log\": true}}'),(141,18,'temp',0.7602407810051485,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 1.5, \"log\": false}}'),(142,18,'arena_temp',0.14313139237609848,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.01, \"high\": 0.5, \"log\": false}}'),(143,18,'cpuct',3.1399134457881286,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 0.5, \"high\": 5.0, \"log\": false}}'),(144,18,'log_epsilon',0.00000009072789538214353,'{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": null, \"low\": 1e-10, \"high\": 1e-07, \"log\": true}}');
/*!40000 ALTER TABLE `trial_params` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trial_system_attributes`
--

DROP TABLE IF EXISTS `trial_system_attributes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trial_system_attributes` (
  `trial_system_attribute_id` int NOT NULL AUTO_INCREMENT,
  `trial_id` int DEFAULT NULL,
  `key` varchar(512) DEFAULT NULL,
  `value_json` text,
  PRIMARY KEY (`trial_system_attribute_id`),
  UNIQUE KEY `trial_id` (`trial_id`,`key`),
  CONSTRAINT `trial_system_attributes_ibfk_1` FOREIGN KEY (`trial_id`) REFERENCES `trials` (`trial_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trial_system_attributes`
--

LOCK TABLES `trial_system_attributes` WRITE;
/*!40000 ALTER TABLE `trial_system_attributes` DISABLE KEYS */;
/*!40000 ALTER TABLE `trial_system_attributes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trial_user_attributes`
--

DROP TABLE IF EXISTS `trial_user_attributes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trial_user_attributes` (
  `trial_user_attribute_id` int NOT NULL AUTO_INCREMENT,
  `trial_id` int DEFAULT NULL,
  `key` varchar(512) DEFAULT NULL,
  `value_json` text,
  PRIMARY KEY (`trial_user_attribute_id`),
  UNIQUE KEY `trial_id` (`trial_id`,`key`),
  CONSTRAINT `trial_user_attributes_ibfk_1` FOREIGN KEY (`trial_id`) REFERENCES `trials` (`trial_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trial_user_attributes`
--

LOCK TABLES `trial_user_attributes` WRITE;
/*!40000 ALTER TABLE `trial_user_attributes` DISABLE KEYS */;
/*!40000 ALTER TABLE `trial_user_attributes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trial_values`
--

DROP TABLE IF EXISTS `trial_values`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trial_values` (
  `trial_value_id` int NOT NULL AUTO_INCREMENT,
  `trial_id` int NOT NULL,
  `objective` int NOT NULL,
  `value` double DEFAULT NULL,
  `value_type` enum('FINITE','INF_POS','INF_NEG') NOT NULL,
  PRIMARY KEY (`trial_value_id`),
  UNIQUE KEY `trial_id` (`trial_id`,`objective`),
  CONSTRAINT `trial_values_ibfk_1` FOREIGN KEY (`trial_id`) REFERENCES `trials` (`trial_id`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trial_values`
--

LOCK TABLES `trial_values` WRITE;
/*!40000 ALTER TABLE `trial_values` DISABLE KEYS */;
INSERT INTO `trial_values` VALUES (1,1,0,0.925,'FINITE'),(2,6,0,0.7649999999999999,'FINITE'),(3,4,0,0.7450000000000001,'FINITE'),(4,5,0,0.78,'FINITE'),(5,8,0,0.835,'FINITE'),(6,7,0,0.96,'FINITE'),(7,9,0,0.7899999999999999,'FINITE'),(8,11,0,0.765,'FINITE'),(9,10,0,0.8400000000000001,'FINITE'),(10,12,0,0.845,'FINITE'),(11,13,0,0.8300000000000001,'FINITE'),(12,16,0,0.9200000000000002,'FINITE'),(13,14,0,0.8550000000000001,'FINITE');
/*!40000 ALTER TABLE `trial_values` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trials`
--

DROP TABLE IF EXISTS `trials`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trials` (
  `trial_id` int NOT NULL AUTO_INCREMENT,
  `number` int DEFAULT NULL,
  `study_id` int DEFAULT NULL,
  `state` enum('RUNNING','COMPLETE','PRUNED','FAIL','WAITING') NOT NULL,
  `datetime_start` datetime DEFAULT NULL,
  `datetime_complete` datetime DEFAULT NULL,
  PRIMARY KEY (`trial_id`),
  KEY `ix_trials_study_id` (`study_id`),
  CONSTRAINT `trials_ibfk_1` FOREIGN KEY (`study_id`) REFERENCES `studies` (`study_id`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trials`
--

LOCK TABLES `trials` WRITE;
/*!40000 ALTER TABLE `trials` DISABLE KEYS */;
INSERT INTO `trials` VALUES (1,0,1,'COMPLETE','2023-09-23 19:48:04','2023-09-23 22:56:35'),(2,1,1,'RUNNING','2023-09-23 19:48:04',NULL),(3,2,1,'RUNNING','2023-09-23 19:48:04',NULL),(4,3,1,'COMPLETE','2023-09-23 19:48:04','2023-09-24 03:17:08'),(5,4,1,'COMPLETE','2023-09-23 19:48:04','2023-09-24 03:35:16'),(6,5,1,'COMPLETE','2023-09-23 22:56:35','2023-09-24 01:22:10'),(7,6,1,'COMPLETE','2023-09-24 01:22:10','2023-09-24 09:27:19'),(8,7,1,'COMPLETE','2023-09-24 03:17:08','2023-09-24 08:31:44'),(9,8,1,'COMPLETE','2023-09-24 03:35:16','2023-09-24 10:53:55'),(10,9,1,'COMPLETE','2023-09-24 08:31:44','2023-09-24 15:34:47'),(11,10,1,'COMPLETE','2023-09-24 09:27:19','2023-09-24 15:25:29'),(12,11,1,'COMPLETE','2023-09-24 10:53:55','2023-09-24 15:53:13'),(13,12,1,'COMPLETE','2023-09-24 15:25:29','2023-09-24 19:37:43'),(14,13,1,'COMPLETE','2023-09-24 15:34:47','2023-09-25 00:01:35'),(15,14,1,'FAIL','2023-09-24 15:53:13','2023-09-25 01:44:33'),(16,15,1,'COMPLETE','2023-09-24 19:37:43','2023-09-24 23:15:05'),(17,16,1,'FAIL','2023-09-24 23:15:05','2023-09-25 01:41:35'),(18,17,1,'FAIL','2023-09-25 00:01:35','2023-09-25 01:39:05');
/*!40000 ALTER TABLE `trials` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `version_info`
--

DROP TABLE IF EXISTS `version_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `version_info` (
  `version_info_id` int NOT NULL,
  `schema_version` int DEFAULT NULL,
  `library_version` varchar(256) DEFAULT NULL,
  PRIMARY KEY (`version_info_id`),
  CONSTRAINT `version_info_chk_1` CHECK ((`version_info_id` = 1))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `version_info`
--

LOCK TABLES `version_info` WRITE;
/*!40000 ALTER TABLE `version_info` DISABLE KEYS */;
INSERT INTO `version_info` VALUES (1,12,'3.2.0');
/*!40000 ALTER TABLE `version_info` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-09-25 11:53:31
