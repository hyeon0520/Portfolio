package com.example.myapplication_project;

import java.util.ArrayList;
import java.util.List;

public class GeminiRequest {
    public List<Content> contents = new ArrayList<>();

    public GeminiRequest(String userText) {
        contents.add(new Content(userText));
    }

    public static class Content {
        public List<Part> parts = new ArrayList<>();
        public Content(String text) {
            parts.add(new Part(text));
        }
    }

    public static class Part {
        public String text;
        public Part(String text) { this.text = text; }
    }
}