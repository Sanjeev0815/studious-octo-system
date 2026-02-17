CREATE DATABASE IF NOT EXISTS expert_discovery;
USE expert_discovery;

CREATE TABLE IF NOT EXISTS users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(64) NOT NULL,
    reputation INT NOT NULL,
    accepted_answers INT NOT NULL,
    total_answers INT NOT NULL,
    activity_score FLOAT NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS questions (
    question_id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    body TEXT NOT NULL,
    tags VARCHAR(255),
    timestamp DATETIME NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS tags (
    tag_id INT PRIMARY KEY AUTO_INCREMENT,
    tag_name VARCHAR(64) NOT NULL UNIQUE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS question_tags (
    question_id INT NOT NULL,
    tag_id INT NOT NULL,
    PRIMARY KEY (question_id, tag_id),
    CONSTRAINT fk_qt_question FOREIGN KEY (question_id) REFERENCES questions(question_id),
    CONSTRAINT fk_qt_tag FOREIGN KEY (tag_id) REFERENCES tags(tag_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS answers (
    answer_id INT PRIMARY KEY AUTO_INCREMENT,
    question_id INT NOT NULL,
    user_id INT NOT NULL,
    answer_text TEXT NOT NULL,
    upvotes INT NOT NULL,
    accepted_flag TINYINT(1) NOT NULL,
    CONSTRAINT fk_answer_question FOREIGN KEY (question_id) REFERENCES questions(question_id),
    CONSTRAINT fk_answer_user FOREIGN KEY (user_id) REFERENCES users(user_id)
) ENGINE=InnoDB;

CREATE INDEX idx_answers_question ON answers(question_id);
CREATE INDEX idx_answers_user ON answers(user_id);
CREATE INDEX idx_questions_timestamp ON questions(timestamp);
