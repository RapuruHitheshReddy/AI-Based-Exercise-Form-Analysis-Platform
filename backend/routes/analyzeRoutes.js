// routes/analyzeRoutes.js
import express from "express";
import { upload } from "../middleware/uploadMiddleware.js";
import { analyzeExercise } from "../controllers/analyzeController.js";

const router = express.Router();

router.post("/analyze", upload.single("video"), analyzeExercise);

export default router;
