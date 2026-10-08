package com.example.myapplication_project;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {

    private GeminiApiService apiService;
    private EditText etWho, etPurpose;
    private TextView tvResult;
    private CardView cardResult;
    private Button btnGenerate;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        etWho = findViewById(R.id.etWho);
        etPurpose = findViewById(R.id.etPurpose);
        tvResult = findViewById(R.id.tvResult);
        cardResult = findViewById(R.id.cardResult);
        btnGenerate = findViewById(R.id.btnGenerate);

        apiService = GeminiApiClient.getApiService();

        Button btnShowQuestions;

        btnShowQuestions = findViewById(R.id.btnShowQuestions);

        btnShowQuestions.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cardResult.setVisibility(View.GONE);
                Intent intent = new Intent(MainActivity.this, QuestionActivity.class);
                startActivity(intent);
            }
        });


        btnGenerate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String who = etWho.getText().toString().trim();
                String purpose = etPurpose.getText().toString().trim();
                String userInput = "만날 사람: " + who + ", 목적/상황: " + purpose + " 대화 주제를 추천해줘.";

                String apiKey = "apikey를 넣으세요";

                GeminiRequest request = new GeminiRequest(userInput);

                Call<GeminiResponse> call = apiService.generateContent(apiKey, request);
                call.enqueue(new Callback<GeminiResponse>() {
                    @Override
                    public void onResponse(Call<GeminiResponse> call, Response<GeminiResponse> response) {
                        if (response.isSuccessful() && response.body() != null && response.body().candidates != null && response.body().candidates.size() > 0) {
                            String result = response.body().candidates.get(0).content.parts.get(0).text;
                            tvResult.setText(result);
                            cardResult.setVisibility(View.VISIBLE);

                            Question q = new Question();
                            q.who = who;
                            q.purpose = purpose;
                            q.userInput = userInput;
                            q.result = result;
                            q.timestamp = System.currentTimeMillis();

                            AppDatabase db = AppDatabase.getInstance(MainActivity.this);
                            db.questionDao().insert(q);
                        } else {
                            String errorMsg = "AI 응답 오류! code=" + response.code();
                            try {
                                if (response.errorBody() != null)
                                    errorMsg += "\n" + response.errorBody().string();
                            } catch (Exception e) {
                                errorMsg += "\n(에러 바디 읽기 실패)";
                            }
                            tvResult.setText(errorMsg);
                            cardResult.setVisibility(View.VISIBLE);
                        }
                    }
                    @Override
                    public void onFailure(Call<GeminiResponse> call, Throwable t) {
                        tvResult.setText("네트워크 오류: " + t.getMessage());
                        cardResult.setVisibility(View.VISIBLE);
                    }
                });
            }
        });
    }
}