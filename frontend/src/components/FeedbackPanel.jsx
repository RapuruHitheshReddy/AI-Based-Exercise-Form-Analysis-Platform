import React from "react";

export default function FeedbackPanel({ summary }) {
  if (!summary) return null;

  const { ai_form_score = 0, total_reps = 0, feedback = [] } = summary;

  // Dynamic color for score ring
  const ringColor =
    ai_form_score >= 80
      ? "text-green-500"
      : ai_form_score >= 60
      ? "text-yellow-500"
      : "text-red-500";

  // Determine feedback color (keywords-based)
  const getFeedbackColor = (text) => {
    const lower = text.toLowerCase();
    if (lower.includes("good") || lower.includes("correct"))
      return "text-green-700";
    if (
      lower.includes("improve") ||
      lower.includes("incorrect") ||
      lower.includes("avoid") ||
      lower.includes("wrong")
    )
      return "text-red-600";
    return "text-gray-700";
  };

  return (
    <div className="relative bg-white/70 backdrop-blur-md border border-blue-100 shadow-lg rounded-3xl p-8 max-w-4xl mx-auto mt-10 transition-all duration-300 hover:shadow-2xl animate-fadeInUp">
      <h2 className="text-3xl font-extrabold text-center text-blue-700 mb-8">
        💡 Form Feedback
      </h2>

      {/* Score + Stats Section */}
      <div className="flex flex-col md:flex-row items-center justify-center gap-10 mb-8">
        {/* Progress Ring */}
        <div className="relative w-36 h-36 flex items-center justify-center">
          <svg className="w-36 h-36 transform -rotate-90">
            <circle
              cx="72"
              cy="72"
              r="60"
              stroke="currentColor"
              className="text-gray-200"
              strokeWidth="10"
              fill="none"
            />
            <circle
              cx="72"
              cy="72"
              r="60"
              stroke="currentColor"
              className={ringColor}
              strokeWidth="10"
              strokeLinecap="round"
              fill="none"
              strokeDasharray={2 * Math.PI * 60}
              strokeDashoffset={
                2 * Math.PI * 60 * (1 - ai_form_score / 100)
              }
            />
          </svg>
          <div className="absolute text-center">
            <p className="text-3xl font-bold text-gray-800">
              {ai_form_score}%
            </p>
            <p className="text-sm text-gray-500 font-medium">Form Score</p>
          </div>
        </div>

        {/* Reps + Details */}
        <div className="text-center md:text-left space-y-3">
          <p className="text-lg text-gray-700 font-semibold">
            🏋️‍♂️ Total Reps:{" "}
            <span className="text-blue-700 font-bold">{total_reps}</span>
          </p>
          <p className="text-md text-gray-600 max-w-sm">
            The AI evaluated your performance using pose estimation and
            biomechanical pattern analysis to generate the form score above.
          </p>
        </div>
      </div>

      {/* Feedback List */}
      <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-2xl p-6 shadow-inner">
        <h3 className="text-xl font-semibold text-blue-800 mb-4 text-center">
          🧠 Actionable Feedback
        </h3>

        {feedback.length > 0 ? (
          <ul className="space-y-3">
            {feedback.map((f, i) => (
              <li
                key={i}
                className={`p-3 rounded-lg border ${
                  getFeedbackColor(f).includes("green")
                    ? "border-green-200 bg-green-50"
                    : getFeedbackColor(f).includes("red")
                    ? "border-red-200 bg-red-50"
                    : "border-gray-200 bg-gray-50"
                } ${getFeedbackColor(f)} shadow-sm`}
              >
                {f}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-center text-gray-500 italic">
            No specific feedback available.
          </p>
        )}
      </div>
    </div>
  );
}
