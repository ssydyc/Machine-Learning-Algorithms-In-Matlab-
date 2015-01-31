function [error_mean,error_std]=naiveBayesDiscrete(data_filename, labels_filename, num_splits, train_percent)
    news_data=csvread(data_filename);
    news_label=csvread(labels_filename);
    doc_length=length(news_label); %how many docments we have
    train_percent=train_percent*0.01;
    results=zeros(num_splits,length(train_percent)); % for each num_splits, we need a vecotr result for each train_percent
    
    for split_k=1:num_splits %do splits for num_splits times
        fprintf('%f percents finished\n',split_k*10);
        test_index=[];
        train_index=[]; %now it's time to get the docID for test documents and train documents
        class_size=zeros(1,20); %class size
        for i=1:doc_length
            class_size(news_label(i))=class_size(news_label(i))+1;
        end
        
        for i=1:20 %for each class
           temp_test_index=sort(randperm(class_size(i),floor(class_size(i)*0.2))); %choose test index
           start=sum(class_size(1:i-1));
           for j=1:class_size(i) %index for each class
               if ismember(j,temp_test_index)
                   test_index=[test_index,start+j];
               else
                   train_index=[train_index,start+j];
               end
           end
        end
        %now we get the train_index and test_index

        
        for i=1:length(train_percent)
           prob_word=ones(20,60000)*0.001+zeros(20,60000);
           prob_class=zeros(1,20); %the probability for each class
           %Now it's time to do training
           map_train=zeros(1,doc_length); %decide whether is training sex
           train_index_i=train_index(sort(randperm(length(train_index),floor(length(train_index)*train_percent(i)))));
           map_train(train_index_i)=1; %now all training_ID docs are mapped into 1
           %get train_index_i for this data_set
           %now it's time to do training!
           for j=1:length(news_data)
               if map_train(news_data(j,1))
                  %it's the tranning data we want, just naive counting
                  temp_class=news_label(news_data(j,1));
                  prob_class(temp_class)=prob_class(temp_class)+news_data(j,3);
                  prob_word(temp_class,news_data(j,2))=prob_word(temp_class,news_data(j,2))+news_data(j,3);
               end
           end
           % we just did counting, now transfer it to probability

           for j=1:20
               prob_word(j,:)=prob_word(j,:)/prob_class(j); %now prob_class is the number of total words in class i
           end
           prob_class=zeros(1,20);
           for j=train_index_i
              prob_class(news_label(j))=prob_class(news_label(j))+1;
           end
           prob_class=prob_class/sum(prob_class);
           %now we get all probability
           
           %Now it's time to do testing,we already have test_index in the begining
           temp_test=0;
           test_prob=[]; %it's time to get test_data
           map_test=zeros(1,doc_length);
           map_test(test_index)=1; %now we have 1 if map_test if it's test data
        
           test_now=0; %we record whether we are currently test data
           for j=1:length(news_data)
                if map_test(news_data(j,1))==1 && test_now==0 % we need to begin test
                    test_now=news_data(j,1); %we are testing document test_now
                    test_prob=log(prob_class); %initialize the probability as Pr(C_k)
                    test_prob=test_prob+news_data(j,3)*log(prob_word(:,news_data(j,2))');
                elseif map_test(news_data(j,1))==0 && test_now==0 % do nothing
                    continue;
                elseif map_test(news_data(j,1))==1 && test_now~=0 && test_now==news_data(j,1) % continue training   
                    test_prob=test_prob+news_data(j,3)*log(prob_word(:,news_data(j,2))');
                elseif map_test(news_data(j,1))==0 && test_now~=0
                    [~,max_index]=max(test_prob);
                    if max_index~=news_label(test_now)
                        temp_test=temp_test+1; %it's wrong, error plus one
                    end
                    test_now=0;
                else % news_data(j,1))==1 && test_now~=0 && test_now~=news_data(j,1), finish previous one and start new one
                    [~,max_index]=max(test_prob);
                    if max_index~=news_label(test_now)
                        temp_test=temp_test+1; %it's wrong, error plus one
                    end
                    test_now=news_data(j,1);
                    test_prob=log(prob_class); %initialize the probability as Pr(C_k)
                    test_prob=test_prob+news_data(j,3)*log(prob_word(:,news_data(j,2))');
                end
           end
           
           if test_now~=0 % we have one more to do
                [~,max_index]=max(test_prob);
                if max_index~=news_label(test_now)
                    temp_test=temp_test+1; %it's wrong, error plus one
                end
           end
           results(split_k,i)=temp_test/length(test_index);     %add one result
           
        end
    end

    error_mean=mean(results);
    error_std=std(results);
    fprintf('percentage:');
    disp(train_percent);
    fprintf('error mean');
    disp(error_mean);
    fprintf('error standard deviation');
    disp(error_std);
end