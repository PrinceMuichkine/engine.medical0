import { Message } from 'ai';

// Configuration for the HealthGPT API
const HEALTHGPT_API_URL = process.env.HEALTHGPT_API_URL || 'http://localhost:5000';

// Interface for message content handling
interface HealthGPTRequestOptions {
  messages: Message[];
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
}

// Type definition for content of a message
type MessageContent = string | { text: string; type?: string } | { image_url: string | { url: string }; type?: string };

/**
 * HealthGPT provider implementation
 */
export const healthgptModel = {
  id: 'healthgpt-model',
  name: 'HealthGPT',
  
  // Generate response (non-streaming)
  async generate(options: HealthGPTRequestOptions): Promise<string> {
    const { messages, maxTokens, temperature } = options;
    
    // Get the last user message which contains either text or image
    const lastUserMessage = messages[messages.length - 1];

    // Check if we have image input
    const imageData = extractImageFromMessage(lastUserMessage);
    const userText = extractTextFromMessage(lastUserMessage);

    if (imageData) {
      // Case 1: We have an image to analyze
      const response = await fetch(`${HEALTHGPT_API_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userText,
          image: imageData,
          model: 'HealthGPT-M3', // Default model, can be parameterized
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HealthGPT API Error: ${error.error || 'Unknown error'}`);
      }

      const data = await response.json();
      return data.answer;
    } else {
      // Case 2: Text-only query, determine if it's for medical image generation
      const isGenerationRequest = shouldGenerateImage(userText);

      if (isGenerationRequest) {
        const response = await fetch(`${HEALTHGPT_API_URL}/api/generate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            prompt: userText,
            model: 'HealthGPT-M3', // Default model
          }),
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(`HealthGPT API Error: ${error.error || 'Unknown error'}`);
        }

        const data = await response.json();
        
        // For image generation, return a message with the image
        return `Here is the generated medical image based on your description: ![Generated medical image](data:image/png;base64,${data.image})`;
      } else {
        // For text-only queries without image generation, pass to another model
        throw new Error('Text-only queries without image analysis/generation are not supported by HealthGPT');
      }
    }
  },

  // Stream response
  async stream(options: HealthGPTRequestOptions): Promise<ReadableStream> {
    const { messages, maxTokens, temperature, signal } = options;
    
    // Use transform streams to handle the response
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    
    const lastUserMessage = messages[messages.length - 1];
    const imageData = extractImageFromMessage(lastUserMessage);
    const userText = extractTextFromMessage(lastUserMessage);

    // Implementation for streaming response
    return new ReadableStream({
      async start(controller) {
        try {
          if (imageData) {
            // Case 1: Medical image analysis
            const response = await fetch(`${HEALTHGPT_API_URL}/api/analyze`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                question: userText,
                image: imageData,
                model: 'HealthGPT-M3',
              }),
              signal,
            });

            if (!response.ok) {
              const error = await response.json();
              throw new Error(`HealthGPT API Error: ${error.error || 'Unknown error'}`);
            }

            const data = await response.json();
            
            // Since the API doesn't support streaming, we'll simulate it
            const chunks = data.answer.split(' ');
            for (const chunk of chunks) {
              controller.enqueue(encoder.encode(chunk + ' '));
              // Small delay to simulate streaming
              await new Promise(resolve => setTimeout(resolve, 10));
            }
          } else {
            // Case 2: Text-based queries for medical image generation
            const isGenerationRequest = shouldGenerateImage(userText);
            
            if (isGenerationRequest) {
              const response = await fetch(`${HEALTHGPT_API_URL}/api/generate`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  prompt: userText,
                  model: 'HealthGPT-M3',
                }),
                signal,
              });
    
              if (!response.ok) {
                const error = await response.json();
                throw new Error(`HealthGPT API Error: ${error.error || 'Unknown error'}`);
              }
    
              const data = await response.json();
              
              // First provide some text
              const text = "Here is the generated medical image based on your description:";
              const chunks = text.split(' ');
              for (const chunk of chunks) {
                controller.enqueue(encoder.encode(chunk + ' '));
                await new Promise(resolve => setTimeout(resolve, 10));
              }
              
              // Then the image markdown
              const imageMarkdown = `\n\n![Generated medical image](data:image/png;base64,${data.image})`;
              controller.enqueue(encoder.encode(imageMarkdown));
            } else {
              controller.enqueue(encoder.encode("Text-only queries without image analysis/generation are not supported by HealthGPT"));
            }
          }
          
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });
  }
};

/**
 * Helper function to extract image data from a message
 */
function extractImageFromMessage(message: Message): string | null {
  if (message.role !== 'user') return null;
  
  // Check if the message has content array with image
  if (Array.isArray(message.content)) {
    // TypeScript needs help here with type narrowing
    const content = message.content as MessageContent[];
    
    const imageContent = content.find(
      (c) => typeof c === 'object' && 'image_url' in c
    );
    
    if (imageContent && typeof imageContent === 'object' && 'image_url' in imageContent) {
      // If URL starts with data:image/*, it's already base64 encoded
      const imageUrl = typeof imageContent.image_url === 'string' 
        ? imageContent.image_url 
        : imageContent.image_url.url;
        
      if (imageUrl.startsWith('data:image')) {
        return imageUrl;
      }
    }
  }
  
  return null;
}

/**
 * Helper function to extract text from a message
 */
function extractTextFromMessage(message: Message): string {
  if (message.role !== 'user') return '';
  
  if (typeof message.content === 'string') {
    return message.content;
  } else if (Array.isArray(message.content)) {
    // TypeScript needs help here with type narrowing
    const content = message.content as MessageContent[];
    
    // Extract text parts from content array
    return content
      .filter((c): c is { text: string; type?: string } => 
        typeof c === 'object' && 'text' in c)
      .map((c) => c.text)
      .join('\n');
  }
  
  return '';
}

/**
 * Helper function to determine if a request is for image generation
 */
function shouldGenerateImage(text: string): boolean {
  // Simple heuristic to detect if the user is asking for image generation
  const generationPhrases = [
    'generate',
    'create',
    'make',
    'draw',
    'visualize',
    'show me',
    'produce',
    'create an image',
    'generate a picture',
    'create a visual',
  ];
  
  const medicalImageTerms = [
    'x-ray',
    'mri',
    'ct scan',
    'ultrasound',
    'radiograph',
    'medical image',
    'scan',
    'anatomy',
    'medical illustration',
    'pathology',
  ];
  
  const hasGenerationPhrase = generationPhrases.some(phrase => 
    text.toLowerCase().includes(phrase)
  );
  
  const hasMedicalImageTerm = medicalImageTerms.some(term => 
    text.toLowerCase().includes(term)
  );
  
  return hasGenerationPhrase && hasMedicalImageTerm;
}

/**
 * Custom provider function to create a HealthGPT provider
 */
export const healthgpt = {
  languageModel: () => healthgptModel,
}; 