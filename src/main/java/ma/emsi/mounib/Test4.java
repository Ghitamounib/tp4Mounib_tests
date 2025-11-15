package ma.emsi.mounib;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;



public class Test4 {
    public static void main(String[] args) {


        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        Path ragPdfPath = Paths.get("src/main/resources/rag.pdf");
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document ragDocument = FileSystemDocumentLoader.loadDocument(ragPdfPath, parser);

        DocumentSplitter documentSplitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> ragSegments = documentSplitter.split(ragDocument);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> embeddingResponse = embeddingModel.embedAll(ragSegments);
        List<Embedding> embeddings = embeddingResponse.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, ragSegments);

        ContentRetriever ragRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        class QueryRouterPourEviterIa implements QueryRouter {

            private final ChatModel chatModel;
            private final ContentRetriever contentRetriever;

            public QueryRouterPourEviterIa(ChatModel chatModel, ContentRetriever contentRetriever) {
                this.chatModel = chatModel;
                this.contentRetriever = contentRetriever;
            }

            @Override
            public List<ContentRetriever> route(Query query) {
                String verificationQuestion =
                        "Est-ce que la requête '" + query.text() + "' porte sur l'IA ? Réponds seulement par 'oui', 'non', ou 'peut-être'.";

                String verificationResponse = chatModel.chat(verificationQuestion);

                if (verificationResponse.toLowerCase().contains("non")) {
                    return Collections.emptyList();
                } else {
                    return Collections.singletonList(contentRetriever);
                }
            }
        }
        QueryRouter queryRouter = new QueryRouterPourEviterIa(chatModel, ragRetriever);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant aiAssistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        try (Scanner inputScanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String userQuestion = inputScanner.nextLine();
                if (userQuestion.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(userQuestion)) {
                    break;
                }
                String assistantResponse = aiAssistant.chat(userQuestion);
                System.out.println("Assistant : " + assistantResponse);
                System.out.println("==================================================");
            }
        }
    }
}
