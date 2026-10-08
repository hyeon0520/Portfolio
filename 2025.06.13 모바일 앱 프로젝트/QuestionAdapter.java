package com.example.myapplication_project;

import android.os.Bundle;
import android.widget.Button;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class QuestionActivity extends AppCompatActivity {

    private RecyclerView recyclerQuestions;
    private Button btnGuess;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_question);

        btnGuess = findViewById(R.id.btnGuessProfile);

        recyclerQuestions = findViewById(R.id.recyclerQuestions);
        recyclerQuestions.setLayoutManager(new LinearLayoutManager(this));

        AppDatabase db = AppDatabase.getInstance(this);
        List<Question> questions = db.questionDao().getAllQuestions();

        QuestionAdapter adapter = new QuestionAdapter(questions);
        recyclerQuestions.setAdapter(adapter);

        Button btnDeleteAll = findViewById(R.id.btnDeleteAll);
        btnDeleteAll.setOnClickListener(v -> {
            new AlertDialog.Builder(this)
                    .setTitle("기록 삭제")
                    .setMessage("정말 모든 기록을 삭제하시겠습니까?")
                    .setPositiveButton("삭제", (dialog, which) -> {
                        db.questionDao().deleteAll();
                        questions.clear();
                        adapter.notifyDataSetChanged();
                    })
                    .setNegativeButton("취소", null)
                    .show();
        });

        btnGuess.setOnClickListener(v -> {
            StringBuilder sb = new StringBuilder();
            for (Question q : questions) {
                if (q.userInput != null) {
                    sb.append("- ").append(q.userInput).append("\n");
                }
            }

            String prompt = "다음은 어떤 사용자가 AI에게 했던 질문 리스트야.\n" +
                    "이 사람의 나이대나 직업을 AI 입장에서 재미로 예측해줘. 너무 딱딱하게 말하지 말고 편하게 얘기해줘.\n\n" +
                    sb.toString();

            GeminiRequest req = new GeminiRequest(prompt);
            GeminiApiService api = GeminiApiClient.getApiService();
            String apiKey = "apikey를 입력하세요";

            api.generateContent(apiKey, req).enqueue(new Callback<GeminiResponse>() {
                @Override
                public void onResponse(Call<GeminiResponse> call, Response<GeminiResponse> response) {
                    String result = "추정 실패...";
                    if (response.isSuccessful() && response.body() != null && response.body().candidates != null
                            && !response.body().candidates.isEmpty()) {
                        result = response.body().candidates.get(0).content.parts.get(0).text;
                    }

                    new AlertDialog.Builder(QuestionActivity.this)
                            .setTitle("AI가 예측한 당신의 정체는?")
                            .setMessage(result)
                            .setPositiveButton("닫기", null)
                            .show();
                }

                @Override
                public void onFailure(Call<GeminiResponse> call, Throwable t) {
                    new AlertDialog.Builder(QuestionActivity.this)
                            .setTitle("에러")
                            .setMessage("예측 실패: " + t.getMessage())
                            .setPositiveButton("닫기", null)
                            .show();
                }
            });
        });
    }
}