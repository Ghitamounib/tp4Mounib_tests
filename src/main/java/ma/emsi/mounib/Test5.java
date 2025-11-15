package ma.emsi.mounib;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;


import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class Test5 {

    public static void main(String[] args) {
        // Charger la clé API Gemini depuis l'environnement
        String geminiApiKey = System.getenv("GEMINI_KEY");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        // Charger et parser le document RAG
        Path ragDocumentPath = Paths.get("src/main/resources/rag.pdf");
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document ragDocument = FileSystemDocumentLoader.loadDocument(ragDocumentPath, documentParser);

        // Split en segments
        DocumentSplitter documentSplitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> textSegments = documentSplitter.split(ragDocument);

        // Modèle d'embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> segmentEmbeddings = embeddingModel.embedAll(textSegments).content();

        // Stockage des embeddings
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(segmentEmbeddings, textSegments);

        // Récupérateur de contenu depuis le PDF
        ContentRetriever pdfContentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Web search avec Tavily
        String tavilyApiKey = System.getenv("TAVILY_KEY");
        WebSearchEngine tavilyApi = TavilyWebSearchEngine.builder()
                .apiKey(tavilyApiKey)
                .build();
        ContentRetriever webContentRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(tavilyApi)
                .build();

        // Routeur de requêtes
        QueryRouter queryRouter = new DefaultQueryRouter(
                pdfContentRetriever,
                webContentRetriever
        );

        // RAG augmentor
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Construire l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // Interaction utilisateur
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String questionUtilisateur = scanner.nextLine();
                if (questionUtilisateur.isBlank()) {
                    continue;
                }
                if ("fin".equalsIgnoreCase(questionUtilisateur)) {
                    break;
                }
                System.out.println("==================================================");
                String reponseAssistant = assistant.chat(questionUtilisateur);
                System.out.println("Assistant : " + reponseAssistant);
                System.out.println("==================================================");
            }
        }
    }
}
