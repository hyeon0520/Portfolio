package com.example.myapplication_project;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "questions")
public class Question {

    @PrimaryKey(autoGenerate = true)
    public int id;

    public String who;
    public String purpose;
    public String userInput;
    public String result;
    public long timestamp; // 저장 시각 (ms)
}
