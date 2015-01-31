function mysgdsvm(filename, k, numruns)
    % numruns should be 5 and k is the number of data to choose
    if strcmp(filename,'MNIST-13.csv')
        MNIST=csvread(filename);
        y=(MNIST(:,1))'-2; % class 1,3 to class -1,1
        feature_size=length(MNIST(1,:))-1;
        data_size=length(MNIST(:,1));
        C1=find(y==-1); %record the index for class -1
        C2=find(y==1); %record the index for class 1
        percent=k/data_size; %the percent of random data
        total_time=zeros(1,numruns);
        w=zeros(1,feature_size);
        for i=1:numruns
            tStart=tic; %record the time
            result=zeros(1,10000); %record the results for plot
            prediction=zeros(1,data_size); %prediction for each step
            for j=1:10000
               size_C1=round(percent*length(C1));
               size_C2=round(percent*length(C2));
               C1_sample=C1(randperm(length(C1),size_C1)); %index of C1 and C2 sample
               C2_sample=C2(randperm(length(C2),size_C2));
               %now we need to have the one with wrong y_i*f(x)<1
               C1_prediction=(prediction(C1_sample))*(-1); % for class -1
               C2_prediction=(prediction(C1_sample));
               A_star1=C1_sample(find(C1_prediction<1));
               A_star2=C2_sample(find(C2_prediction<1)); % index that's in A*
               A_star=[A_star1,A_star2];
               step_t=1/j; %stemp size
               w=(1-step_t)*w+step_t/(size_C1+size_C2)*y(A_star)*MNIST(A_star,2:end);
               if norm(w)>1
                    w=w/norm(w);
               end
               prediction=(MNIST(:,2:end)*w')';
               result(j)=1/2*(norm(w))^2+mean(max(0,1-y.*prediction));
              
            end
            total_time(i)=toc(tStart); %record the time
            X=[1:10,11:10:100,100:100:1000,1000:1000:10000];
            plot(X,result(X));
            hold on;
        end
        xlabel('iteration')
        ylabel('objective function')
        fprintf('total mean runtime %f seconds, standard deviation %f seconds \n',[mean(total_time),std(total_time)]);
    end
end